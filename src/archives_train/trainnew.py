# date: 9/16/2026
# Function: rewriting the train stuff to be more... cohesive.

import src.trainfunc as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score, recall_score

# load the processed data
X, y, participant_ids = tf.load_processed_data("data/processed_features.npz")

# split into test & train
X_train, X_test, y_train, y_test, train_participants, test_participants = tf.split_by_participant(X, y, participant_ids)

# initialize model
model = RandomForestClassifier(
    n_estimators = 500,
    min_samples_leaf = 20,
    random_state = 42, 
    n_jobs = -1 # means use all available CPU cores
)

# train & test
model, y_train_pred, y_test_pred = tf.train_and_test_model(model, X_train, y_train, X_test, y_test)

# print emotion score counts
tf.print_label_counts(y_test, "Test")
tf.print_label_counts(y_train, "Train")

# prediction accuracy, by emotion score, for test
tf.pred_label_accuracy(y_test, y_test_pred)
