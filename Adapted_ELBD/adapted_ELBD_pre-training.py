# This code is adapted from https://github.com/AXinx/ELBD

from __future__ import division
from __future__ import print_function

from time import time
import warnings
# Supress warnings for clean output.
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.utils.utility import standardizer

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics

from keras import models
from keras import layers


labeled_data = pd.read_pickle('labeled_data.pkl')

# Normalize the data.
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(labeled_data.drop(['timestamp'], axis=1)))
normalized_data.columns = labeled_data.columns[1:]
normalized_labeled_data = pd.merge(labeled_data.timestamp, normalized_data, left_index=True, right_index=True, how='left')
normalized_labeled_data.fillna(method='ffill', inplace=True)  # Forward fill.
normalized_labeled_data.fillna(method='bfill', inplace=True)  # Backward fill any remaining NaNs.

def pred_labels(y_test, test_scores):
    fpr, tpr, thresholds = roc_curve(y_test, test_scores)
    cutoff = thresholds[np.argmax(tpr - fpr)]
    pred_label = []
    for each in test_scores:
        if each > cutoff:
            pred_label.append(1)
        else:
            pred_label.append(0)    
    return pred_label


def cal_eval(label, pred_label, scores):
    pr = round(metrics.precision_score(label, pred_label, average='macro'), ndigits=4)
    re = round(metrics.recall_score(label, pred_label, average='macro'), ndigits=4)
    f1 = round(metrics.f1_score(label, pred_label, average='macro'), ndigits=4)
    roc = round(roc_auc_score(label, scores), ndigits=4)
    return pr, re, f1, roc


X_train = normalized_labeled_data.iloc[:, 1:]
y_train = normalized_labeled_data['label']
X_test = normalized_labeled_data.iloc[:, 1:]
y_test = normalized_labeled_data['label']

outliers_fraction = np.count_nonzero(y_train) / len(y_train)  
random_state = np.random.RandomState(42)
classifiers = {
        'Isolation Forest': IForest(contamination=outliers_fraction,
                                    random_state=random_state),
        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
        'Local Outlier Factor (LOF)': LOF(
            contamination=outliers_fraction),
        'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction)
}

n_clf = len(classifiers)
train_scores = np.zeros([X_train.shape[0], n_clf])
test_scores = np.zeros([X_test.shape[0], n_clf])
p_labels = np.zeros([X_test.shape[0], n_clf])

i = 0
train_duration = 0
test_duration = 0
for clf_name, clf in classifiers.items():
    t0 = time()
    clf.fit(X_train)
    t1 = time()
    test_sccore = clf.decision_function(X_test)
    t_t = time()
    test_mean_score = np.nanmean(test_scores)
    test_scores[np.isnan(test_scores)] = test_mean_score
    test_scores[:, i] = test_sccore
    p_labels[:, i] = pred_labels(y_test, test_sccore)
    i += 1
    print('Base detector %i is fitted for prediction' % i)
    train_duration += round(t1 - t0, ndigits=4)
    test_duration += round(t_t - t1, ndigits=4)
    
# Standardize test scores.
test_scores_norm = standardizer(test_scores)

train_x, test_x, train_y, test_y = train_test_split(test_scores_norm, y_test, test_size=0.9, random_state=random_state)

def ensemble_mlp(df_columns):
    enm_mlp_df = pd.DataFrame(columns=df_columns)

    t2 = time()
    model = models.Sequential()
    # Input Layer
    model.add(layers.Dense(4, activation = "relu", input_shape=(len(train_y), 4)))
    model.add(layers.Dense(20, activation = "relu"))
    model.add(layers.Dense(20, activation = "relu"))
    # Output Layer
    model.add(layers.Dense(1, activation = "sigmoid"))
    model.summary()
    # Compile the model.
    model.compile(
        optimizer = "adam",
        loss = "binary_crossentropy",
        metrics = ["accuracy"]
    )
    model.fit(
     train_x, train_y,
     epochs = 100,
     batch_size = 20,
     validation_data = (train_x, train_y)
    )
    t3 = time()
    pred_test = model.predict(test_scores_norm)
    t4 = time()
    train_time = t3 - t2
    test_time = t4 - t3
    train_time_mlp = round(train_duration + train_time, ndigits=4)
    test_time_mlp = round(test_duration + test_time, ndigits=4)
    
    pred_nn = []
    for each in pred_test:
        if each[0] > 0.5:
            pred_nn.append(1)
        else:
            pred_nn.append(0)
    pr_mlp, re_mlp, f1_mlp, roc_mlp = cal_eval(y_test, pred_nn, pred_test)
    enm_mlp = pd.DataFrame([pr_mlp, re_mlp, f1_mlp, roc_mlp, train_time_mlp, test_time_mlp]).transpose()
    enm_mlp.columns = df_columns
    enm_mlp_df = pd.concat([enm_mlp_df, enm_mlp], axis=0)
    return pred_nn, enm_mlp_df, model


df_columns = ['Precision', 'Recall', 'F1 score', 'AUC', 'Train time', 'Test time']
pred_nn, enm_mlp_df, model = ensemble_mlp(df_columns)
# Display the evaluation results.
print(enm_mlp_df)
# Save the trained model.
model.save('pre-trained_ELBD.keras')
