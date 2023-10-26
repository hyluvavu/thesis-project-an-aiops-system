#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This code is adapted from https://github.com/AXinx/CausalRCA_code

import warnings
warnings.filterwarnings('ignore')

import time
import math
import requests

import numpy as np
import pandas as pd
# import networkx as nx
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify

from utils import *
from modules import *
from config import CONFIG

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS


def fetch_metrics(start_time, end_time, step='5s', microservices=['carts', 'catalogue', 'front-end', 'orders', 'payment', 'shipping', 'user']):
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


def get_relevant_data(data, relevance):
    """
    Extract data related to a specific microservice or metric
    from the given metrics dataframe.
    
    Parameters:
    - data: Dataframe containing the metrics data
    - relevance: String indicating the name of a microservice or metric
    
    Returns:
    - Dataframe containing only the relevant data
    """
        
    relevant_cols = [col for col in data.columns if relevance in col]
    relevant_data = data[relevant_cols].copy()
    
    return relevant_data


def perform_rca(data):
    """
    Perform root cause analysis given the metrics data.
    
    Parameters:
    - data: Dataframe containing the metrics data
    
    Returns:
    - List of possible root causes ranked by their PageRank scores
    """
    
    CONFIG.cuda = torch.cuda.is_available()
    CONFIG.factor = not CONFIG.no_factor

    # Store column names for the purpose of ranking them in the end.
    column_names = data.columns
    
    data_sample_size = data.shape[0]
    data_variable_size = data.shape[1]
    
    
    # ======================================================
    # Get Data: experiments = {synthetic SEM, ALARM}
    # ======================================================
    
    train_data = data
    
    
    # ======================================================
    # Load Modules
    # ======================================================
    
    # Generate a off-diagonal interaction graph.
    off_diag = np.ones([data_variable_size, data_variable_size]) - np.eye(data_variable_size)

    # Add adjacency matrix A.
    num_nodes = data_variable_size
    adj_A = np.zeros((num_nodes, num_nodes))


    if CONFIG.encoder == 'mlp':
        encoder = MLPEncoder(data_variable_size * CONFIG.x_dims, CONFIG.x_dims, CONFIG.encoder_hidden,
                             int(CONFIG.z_dims), adj_A,
                             batch_size = CONFIG.batch_size,
                             do_prob = CONFIG.encoder_dropout, factor = CONFIG.factor).double()
    elif CONFIG.encoder == 'sem':
        encoder = SEMEncoder(data_variable_size * CONFIG.x_dims, CONFIG.encoder_hidden,
                             int(CONFIG.z_dims), adj_A,
                             batch_size = CONFIG.batch_size,
                             do_prob = CONFIG.encoder_dropout, factor = CONFIG.factor).double()

    if CONFIG.decoder == 'mlp':
        decoder = MLPDecoder(data_variable_size * CONFIG.x_dims,
                             CONFIG.z_dims, CONFIG.x_dims, encoder,
                             data_variable_size = data_variable_size,
                             batch_size = CONFIG.batch_size,
                             n_hid = CONFIG.decoder_hidden,
                             do_prob = CONFIG.decoder_dropout).double()
    elif CONFIG.decoder == 'sem':
        decoder = SEMDecoder(data_variable_size * CONFIG.x_dims,
                             CONFIG.z_dims, 2, encoder,
                             data_variable_size = data_variable_size,
                             batch_size = CONFIG.batch_size,
                             n_hid = CONFIG.decoder_hidden,
                             do_prob = CONFIG.decoder_dropout).double()

        
    # ======================================================
    # Set up Training Parameters
    # ======================================================
    
    if CONFIG.optimizer == 'Adam':
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=CONFIG.lr)
    elif CONFIG.optimizer == 'LBFGS':
        optimizer = optim.LBFGS(list(encoder.parameters()) + list(decoder.parameters()),
                               lr=CONFIG.lr)
    elif CONFIG.optimizer == 'SGD':
        optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()),
                               lr=CONFIG.lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=CONFIG.lr_decay,
                                    gamma=CONFIG.gamma)

    # Linear indices of an upper triangular mx, used for acc calculation
    triu_indices = get_triu_offdiag_indices(data_variable_size)
    tril_indices = get_tril_offdiag_indices(data_variable_size)

    if CONFIG.prior:
        prior = np.array([0.91, 0.03, 0.03, 0.03])  # hard coded for now
        print("Using prior")  
        print(prior)
        log_prior = torch.DoubleTensor(np.log(prior))
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = torch.unsqueeze(log_prior, 0)
        log_prior = Variable(log_prior)

        if CONFIG.cuda:
            log_prior = log_prior.cuda()

    if CONFIG.cuda:
        encoder.cuda()
        decoder.cuda()
        triu_indices = triu_indices.cuda()
        tril_indices = tril_indices.cuda()

    # Compute constraint h(A) value.
    def _h_A(A, m):
        expm_A = matrix_poly(A * A, m)
        h_A = torch.trace(expm_A) - m
        return h_A

    prox_plus = torch.nn.Threshold(0., 0.)

    def stau(w, tau):
        w1 = prox_plus(torch.abs(w) - tau)
        return torch.sign(w) * w1

    def update_optimizer(optimizer, original_lr, c_A):
        '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
        MAX_LR = 1e-2
        MIN_LR = 1e-4

        estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
        if estimated_lr > MAX_LR:
            lr = MAX_LR
        elif estimated_lr < MIN_LR:
            lr = MIN_LR
        else:
            lr = estimated_lr

        # set LR
        for parame_group in optimizer.param_groups:
            parame_group['lr'] = lr

        return optimizer, lr

    
    # ======================================================
    # Train
    # ======================================================
    
    def train(epoch, best_val_loss, lambda_A, c_A, optimizer):
        t = time.time()
        nll_train = []
        kl_train = []
        mse_train = []

        encoder.train()
        decoder.train()
        scheduler.step()

        # update optimizer
        optimizer, lr = update_optimizer(optimizer, CONFIG.lr, c_A)

        for i in range(1):
            data = train_data[i * data_sample_size : (i + 1) * data_sample_size]
            data = torch.tensor(data.to_numpy().reshape(data_sample_size, data_variable_size, 1))
            if CONFIG.cuda:
                data = data.cuda()
            data = Variable(data).double()

            optimizer.zero_grad()

            enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data)  # logits is of size: [num_sims, z_dims]
            edges = logits
            dec_x, output, adj_A_tilt_decoder = decoder(data, edges, data_variable_size * CONFIG.x_dims, origin_A, adj_A_tilt_encoder, Wa)

            if torch.sum(output != output):
                print('nan error\n')

            target = data
            preds = output
            variance = 0.

            # Reconstruction accuracy loss
            loss_nll = nll_gaussian(preds, target, variance)

            # KL loss
            loss_kl = kl_gaussian_sem(logits)

            # ELBO loss
            loss = loss_kl + loss_nll
            
            # Add A loss.
            one_adj_A = origin_A  # torch.mean(adj_A_tilt_decoder, dim =0)
            sparse_loss = CONFIG.tau_A * torch.sum(torch.abs(one_adj_A))

            # Other loss terms
            if CONFIG.use_A_connect_loss:
                connect_gap = A_connect_loss(one_adj_A, CONFIG.graph_threshold, z_gap)
                loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

            if CONFIG.use_A_positiver_loss:
                positive_gap = A_positive_loss(one_adj_A, z_positive)
                loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

            # Compute h(A).
            h_A = _h_A(origin_A, data_variable_size)
            loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A * origin_A) + sparse_loss

            loss.backward()
            loss = optimizer.step()

            myA.data = stau(myA.data, CONFIG.tau_A*lr)

            if torch.sum(origin_A != origin_A):
                print('nan error\n')

            # Compute metrics.
            graph = origin_A.data.clone().cpu().numpy()
            graph[np.abs(graph) < CONFIG.graph_threshold] = 0

            mse_train.append(F.mse_loss(preds, target).item())
            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())

        return np.mean(np.mean(kl_train) + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A

    
    # ======================================================
    # Main
    # ======================================================

    # gamma = args.gamma
    # eta = args.eta
    gamma = 0.25
    eta = 10

    t_total = time.time()
    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    best_epoch = 0
    best_ELBO_graph = []
    best_NLL_graph = []
    best_MSE_graph = []
    # Optimizer step on hyparameters
    c_A = CONFIG.c_A
    lambda_A = CONFIG.lambda_A
    h_A_new = torch.tensor(1.)
    h_tol = CONFIG.h_tol
    k_max_iter = int(CONFIG.k_max_iter)
    h_A_old = np.inf

    E_loss = []
    N_loss = []
    M_loss = []
    start_time = time.time()
    try:
        for step_k in range(k_max_iter):
            while c_A < 1e+20:
                for epoch in range(CONFIG.epochs):
                    ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = train(epoch, best_ELBO_loss, lambda_A, c_A, optimizer)
                    E_loss.append(ELBO_loss)
                    N_loss.append(NLL_loss)
                    M_loss.append(MSE_loss)
                    if ELBO_loss < best_ELBO_loss:
                        best_ELBO_loss = ELBO_loss
                        best_epoch = epoch
                        best_ELBO_graph = graph

                    if NLL_loss < best_NLL_loss:
                        best_NLL_loss = NLL_loss
                        best_epoch = epoch
                        best_NLL_graph = graph

                    if MSE_loss < best_MSE_loss:
                        best_MSE_loss = MSE_loss
                        best_epoch = epoch
                        best_MSE_graph = graph

                if ELBO_loss > 2 * best_ELBO_loss:
                    break

                # Update parameters.
                A_new = origin_A.data.clone()
                h_A_new = _h_A(A_new, data_variable_size)
                if h_A_new.item() > gamma * h_A_old:
                    c_A *= eta
                else:
                    break

            # Update parameters.
            # h_A, adj_A are computed in loss anyway, so no need to store
            h_A_old = h_A_new.item()
            lambda_A += c_A * h_A_new.item()

            if h_A_new.item() <= h_tol:
                break

        graph = origin_A.data.clone().cpu().numpy()
        graph[np.abs(graph) < 0.1] = 0
        graph[np.abs(graph) < 0.2] = 0
        graph[np.abs(graph) < 0.3] = 0

    except KeyboardInterrupt:
        print('Done!')

    end_time = time.time()
    adj = graph
    # Comment out the following lines to disable graphical output.
    # org_G = nx.from_numpy_matrix(adj, parallel_edges=True, create_using=nx.DiGraph)
    # pos = nx.circular_layout(org_G)
    # nx.draw(org_G, pos=pos, with_labels=True)

    # PageRank
    from sknetwork.ranking import PageRank
    pagerank = PageRank()
    scores = pagerank.fit_transform(np.abs(adj.T))

    score_dict = {}
    for i,s in enumerate(scores):
        score_dict[i] = s

    # Rank microservices/metrics by their PageRank scores.
    ranking_by_score = sorted(score_dict.items(), key=lambda item:item[1], reverse=True)
    ranked_results = [(column_names[tup[0]], tup[1]) for tup in ranking_by_score]
    
    return ranked_results


# ================================================
# Store the Results of CausalRCA to InfluxDB
# ================================================

def write_to_influxdb(start_time, ranked_microservices, granularity):
    """
    Write the ranked results of CausalRCA to InfluxDB.

    Parameters:
    - start_time: The starting timestamp of the time range
    - ranked_microservices: List of possible root causes ranked with PageRank
    - granularity: String with value of 'coarse' or 'fine',
    indicating the granularity of an root cause analysis
    """
    
    # URL of the InfluxDB instance
    # URL = 'http://localhost:8086'
    
    # The name of the InfluxDB Service is 'seal-influxdb2'.
    URL = 'http://seal-influxdb2:80'

    # InfluxDB operator API token created when setting up InfluxDB
    TOKEN = '8VOesz5g8IG5FMKoaa9gycfFUuANahDHYTqe5-bV-bCKfdCk9tf5Hfge5RxisvM4E1XBTSsJbcMeVcEjsPd8tw=='
    ORG = 'influxdata'
    BUCKET = 'CausalRCA'

    try:
        # Initialize a client.
        client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
        write_api = client.write_api(write_options=SYNCHRONOUS)

        for _, value in enumerate(ranked_microservices):        
            data_point = Point('causalrca_results') \
                .tag('anomaly_time', start_time) \
                .tag('granularity', granularity) \
                .field('metric', value[0]) \
                .field('score', value[1])

            write_api.write(bucket=BUCKET, org=ORG, record=data_point)
        
        print('Results written to InfluxDB successfully!')
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Close the client.
        client.close()


# ======================================================
# Set up an HTTP Server to Listen for Incoming Requests
# ======================================================

app = Flask(__name__)
received_data = dict()

@app.route('/', methods=['POST'])
def get_data_from_elbd():
    global received_data
    received_data = request.json
    
    return jsonify({"message": "Data received and processed!"})


@app.after_request
def run_after_request(response):
    query_range_list = received_data.get('rca_query_ranges', [])
    if query_range_list == []:
        print('No anomalies detected and no root cause analysis performed!')
    else:
        for query_range_dict in query_range_list:
            # Convert the ISO format strings to strings indicating UTC
            # for the purpose of querying Prometheus.
            start_time = query_range_dict['start_time'] + 'Z'
            end_time = query_range_dict['end_time'] + 'Z'

            fetched_metrics = fetch_metrics(start_time, end_time)
            normalized_data = normalize_data(fetched_metrics)
            
            # Localize the most possible faulty microservice.
            print('************************************************************')
            print(f'For the time range ({start_time}, {end_time}):')
            print('************************************************************')
            print('Localizing the most possible faulty microservice...')
            response_time = get_relevant_data(normalized_data, 'response_time')
            ranked_microservices = perform_rca(response_time)
            print('The most possible faulty microservices are:')
            print(ranked_microservices)
            write_to_influxdb(start_time, ranked_microservices, 'coarse')
            print('************************************************************')
            
            # Find the most possible root cause metrics of this faulty microservice.
            faulty_microservice = ranked_microservices[0][0].replace('_response_time', '')
            print(f'Finding possible root cause metrics of the most possible faulty microservice ({faulty_microservice})...')
            input_metrics = get_relevant_data(normalized_data, faulty_microservice)
            ranked_metrics = perform_rca(input_metrics)
            print('The most possible root cause metrics are:')
            print(ranked_metrics)
            write_to_influxdb(start_time, ranked_metrics, 'fine')
            print('************************************************************')
            print()
    
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)