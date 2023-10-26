# This code is adapted from https://github.com/AXinx/ELBD

from __future__ import division
from __future__ import print_function

import warnings
# Supress warnings for clean output.
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests

from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.utils.utility import standardizer

from sklearn.preprocessing import MinMaxScaler
from keras import models

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS


# =========================================================
# Fetch Metrics from Prometheus
# =========================================================

def fetch_metrics(start_time, end_time, step, microservices):
    """
    Fetch metrics for a list of microservices from Prometheus
    within a specified time range.
    
    Parameters:
    - start_time: The starting timestamp of the time range
    - end_time: The ending timestamp of the time range
    - step: Time step granularity of the query (e.g., '5s' for 5 seconds)
    - microservices: List of names of the microservices for which metrics are to be fetched
    
    Returns:
    - Dataframe containing the fetched metrics, indexed by timestamps
    """

    # Define the Prometheus endpoint.
    # PROMETHEUS_URL = 'http://localhost:9090/api/v1/query_range'

    # The name of the Prometheus Service is 'octopus-kube-prometheus-st-prometheus'.
    PROMETHEUS_URL = 'http://octopus-kube-prometheus-st-prometheus:9090/api/v1/query_range'
    
    fetched_metrics = None

    for microservice in microservices:
        
        # Define the query for the response time.
        response_time_query = f"""
        sum(
            rate(
                istio_request_duration_milliseconds_sum{{
                    destination_service_name="{microservice}"
                }}[1m]
            )
        )
        /
        sum(
            rate(
                istio_request_duration_milliseconds_count{{
                    destination_service_name="{microservice}"
                }}[1m]
            )
        )
        """

        # Define the query for the CPU usage in the last minute.
        cpu_usage_query = f"""
        sum(
            rate(
                container_cpu_usage_seconds_total{{
                    container="{microservice}"
                }}[1m]
            )
        ) by (pod)
        """
        
        # Define the query for the total memory usage in bytes.
        memory_usage_query = f"""
        sum(
            container_memory_working_set_bytes{{
                container="{microservice}"
            }}
        ) by (pod)
        """

        # Define the query for the bytes received over the network
        # by the container per second in the last minute.
        network_receive_query = f"""
        sum(
            rate(
                container_network_receive_bytes_total{{
                    namespace="sock-shop"
                }}[1m]
            )
            * on(namespace, pod)
            group_left(workload)
            namespace_workload_pod:kube_pod_owner:relabel{{
                namespace="sock-shop",
                workload="{microservice}"
            }}
        ) by (pod)
        """

        # Define the query for the bytes transmitted over the network
        # by the container per second in the last minute.
        network_transmit_query = f"""
        sum(
            rate(
                container_network_transmit_packets_total{{
                    namespace="sock-shop"
                }}[1m])
            * on (namespace,pod)
            group_left(workload)
            namespace_workload_pod:kube_pod_owner:relabel{{
                namespace="sock-shop",
                workload="{microservice}"
            }}
        ) by (pod)
        """
        queries = {
            'response_time': response_time_query,
            'cpu_usage': cpu_usage_query,
            'memory_usage': memory_usage_query,
            'network_receive': network_receive_query,
            'network_transmit': network_transmit_query
        }

        for metric, query in queries.items():
            # Make the API request.
            response = requests.get(
                PROMETHEUS_URL,
                params={
                    'query': query,
                    'start': start_time,
                    'end': end_time,
                    'step': step
                }
            )
            
            # Check if the request was successful.
            response.raise_for_status()

            # Convert the data to a dataframe.
            data = response.json()['data']['result'][0]['values']
            df = pd.DataFrame(data, columns=['timestamp', f'{microservice}_{metric}'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

            if fetched_metrics is None:
                fetched_metrics = df
            else:
                fetched_metrics = pd.merge(fetched_metrics, df, on='timestamp', how='outer')
    
    return fetched_metrics


# =========================================================
# Preprocess the Data, Load the Model, and Detect Anomalies
# =========================================================

def normalize_data(data):
    """
    Normalize the metrics data using Min-Max scaling.
    
    Parameters:
    - data: Dataframe containing the metrics data,
    where the first column is 'timestamp'
    
    Returns:
    - Dataframe containing the normalized metrics data,
    maintaining the original timestamps
    """

    scaler = MinMaxScaler()
    normalized_data_temp = pd.DataFrame(scaler.fit_transform(data.drop(['timestamp'], axis=1)))
    normalized_data_temp.columns = data.columns[1:]
    normalized_data = pd.merge(data.timestamp, normalized_data_temp, left_index=True, right_index=True, how='left')
    normalized_data.fillna(method='ffill', inplace=True)  # Forward fill.
    normalized_data.fillna(method='bfill', inplace=True)  # Backward fill any remaining NaNs.
    
    return normalized_data


def generate_base_model_scores(normalized_data):
    """
    Generate decision scores for anomalies using base models.
    
    Parameters:
    - normalized_data: Dataframe containing normalized metrics data
    
    Returns:
    - 2D Numpy array containing the decision scores for each base model
    """

    input_data = normalized_data.iloc[:, 1:]
    scores = []
    
    random_state = np.random.RandomState(42)
    classifiers = {
            'Isolation Forest': IForest(random_state=random_state),
            'K Nearest Neighbors (KNN)': KNN(),
            'Local Outlier Factor (LOF)': LOF(),
            'One-class SVM (OCSVM)': OCSVM()
    }
    
    for _, clf in classifiers.items():
        clf.fit(input_data)
        decision_scores = clf.decision_function(input_data)
        scores.append(decision_scores)

    base_model_scores = np.column_stack(scores)
    
    return base_model_scores


def detect_anomalies(fetched_metrics):
    """
    Detect anomalies in fetched metrics
    using an ensemble of base models and a pre-trained MLP.
    
    Parameters:
    - fetched_metrics: Dataframe containing the fetched metrics
    
    Returns:
    - List of predicted labels (1 for anomaly, 0 for normal)
    """

    # Load the pre-trained model.
    model = models.load_model('pre-trained_ELBD.keras')
    model.summary()
    
    normalized_data = normalize_data(fetched_metrics)
    
    # Generate decision scores from base models.
    base_model_scores = generate_base_model_scores(normalized_data)
    
    # Standardize the decision scores.
    standardized_scores = standardizer(base_model_scores)
    
    # Predict using ensemble MLP.
    predictions = model.predict(standardized_scores)
    pred_labels = [1 if p[0] > 0.5 else 0 for p in predictions]
    
    return pred_labels


def group_anomalies(fetched_metrics, pred_labels, window_size=60, anomaly_threshold=24):
    """
    Group anomalies based on the number of anomalies in the sliding window.

    Parameters:
    - fetched_metrics: Dataframe containing the fetched metrics
    - pred_labels: List of predicted labels
    - window_size: Size of the window 
    (e.g., for 5 minutes with 5s step, window_size would be 60)
    - anomaly_threshold: Minimum number of anomalies in the window
    to consider it an anomaly

    Returns:
    - List of (start, end) timestamp pairs indicating anomalies
    """

    # Convert pred_labels to a dataframe.
    pred_labels_df = pd.DataFrame(pred_labels, columns=['pred_label'])
    # Associate timestamps from fetched_metrics with pred_labels.
    result_df = pd.concat([fetched_metrics['timestamp'], pred_labels_df], axis=1)

    anomalies = []
    i = 0
    
    while i < len(result_df):
        window = result_df.iloc[i:i+window_size]
        if window['pred_label'].sum() >= anomaly_threshold:
            start_timestamp = window.iloc[0]['timestamp']
            end_timestamp = window.iloc[-1]['timestamp']
            anomalies.append((start_timestamp, end_timestamp))
            i += window_size  # Move the window forward by its size.
        else:
            i += 1

    return anomalies


# start_time = '2023-09-28T13:25:00Z'
# end_time = '2023-09-28T14:25:00Z'

# Get the current UTC time.
current_time = datetime.utcnow()

# Subtract 1 hour to get the starting time.
one_hour_ago = current_time - timedelta(hours=1)

# Format start_time and end_time as strings.
start_time = one_hour_ago.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'
end_time = current_time.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'

step = '5s'
microservices = ['carts', 'catalogue', 'front-end', 'orders', 'payment', 'shipping', 'user']

fetched_metrics = fetch_metrics(start_time, end_time, step, microservices)
pred_labels = detect_anomalies(fetched_metrics)
anomalies = group_anomalies(fetched_metrics, pred_labels)

if anomalies == []:
    print('There is no anomaly detected at this time!')
else:
    print('Anomalies detected in the following time ranges (UTC):')
    for anomaly in anomalies:
        print(anomaly)


# =========================================================
# Inform CausalRCA of the Detected Anomalies
# =========================================================

def inform_causalrca(anomalies):
    """
    Inform CausalRCA to fetch the metrics data for multiple anomalies
    and to start performing root cause analysis.
    
    Parameters:
    - anomalies: A list of tuples,
    each containing a start and end timestamp for an anomaly
    """
    
    # CausalRCA API endpoint URL
    # CAUSALRCA_API_URL = 'http://127.0.0.1:5000/'
    
    # The name of the CausalRCA Service is 'causalrca-service'.
    CAUSALRCA_API_URL = 'http://causalrca-service:5000'
    
    if anomalies == []:
        data = {'rca_query_ranges': []}
    else:
        # Convert the list of Timestamp objects to a list of ISO format strings,
        # so that it is JSON serializable.
        rca_query_ranges_iso = [
            {'start_time': start.isoformat(), 'end_time': (start + pd.Timedelta(minutes=5)).isoformat()}
            for start, end in anomalies
        ]

        # Payload data to be sent to CausalRCA
        data = {'rca_query_ranges': rca_query_ranges_iso}
    
    try:
        response = requests.post(CAUSALRCA_API_URL, json=data)
        
        # Handle response.
        if response.status_code == 200:
            print(f'Informed CausalRCA successfully!')
            print(f'Response from CausalRCA: {response.text}')
        else:
            print(f'Failed to inform CausalRCA. Status code: {response.status_code}. Response text: {response.text}')

    except requests.RequestException as e:
        print(f'Error occurred while informing CausalRCA: {e}')


inform_causalrca(anomalies)


# =========================================================
# Store the Results of ELBD to InfluxDB
# =========================================================

def write_to_influxdb(fetched_metrics, pred_labels):
    """
    Write the prediction results of ELBD to InfluxDB.

    Parameters:
    - fetched_metrics: Dataframe containing the fetched metrics
    - pred_labels: List of predicted labels (1 for anomaly, 0 for normal)
    """
    
    # URL of the InfluxDB instance
    # URL = 'http://localhost:8086'
    
    # The name of the InfluxDB Service is 'seal-influxdb2'.
    URL = 'http://seal-influxdb2:80'

    # InfluxDB operator API token created when setting up InfluxDB
    TOKEN = '8VOesz5g8IG5FMKoaa9gycfFUuANahDHYTqe5-bV-bCKfdCk9tf5Hfge5RxisvM4E1XBTSsJbcMeVcEjsPd8tw=='
    ORG = 'influxdata'
    BUCKET = 'ELBD'

    try:
        # Initialize a client.
        client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
        write_api = client.write_api(write_options=SYNCHRONOUS)

        for index, value in enumerate(pred_labels):
            # Convert pandas Timestamp to datetime object.
            dt_obj = fetched_metrics.timestamp[index].to_pydatetime()

            data_point = Point('elbd_results') \
                .time(dt_obj) \
                .field('anomaly', int(value))  # 1 for anomaly, 0 for normal
            write_api.write(bucket=BUCKET, org=ORG, record=data_point)

        print('Results written to InfluxDB successfully!')

    except Exception as e:
        print(f'An error occurred: {e}')
    
    finally:
        # Close the client.
        client.close()


write_to_influxdb(fetched_metrics, pred_labels)