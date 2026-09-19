# date: 9/16/2026
# Function: to make actual functions for the training so its not... really messy

## --- LOAD PROCESSED DATA --- ##
# parameter: file_path = "data/processed_features.npz"
def load_processed_data(file_path):
    data = np.load(file_path)

    X = data["dataset_features"]
    y = data["y"]
    participant_ids = data["participant_ids"]

    return X, y, participant_ids

## --- SPLITTING PARTICIPANTS INTO TEST & TRAIN --- ##
import numpy as np
from sklearn.model_selection import train_test_split

def split_by_participant(X, y, participant_ids, test_size=0.2, random_state=42):
    unique_participants = np.unique(participant_ids)

    # partition the unique participants into training and testing sets
    train_participants, test_participants = train_test_split(
        unique_participants,
        test_size=test_size, # 0-1, determines percentage of participants used for testing
        random_state=random_state # fixed starting point for random selection; every time get same participants in training/testing
    )

    # mask = an array of True and False values that acts like a filter
    # assigns true/false to each of 21,298 windows.  
    train_mask = np.isin(participant_ids, train_participants)
    test_mask = np.isin(participant_ids, test_participants)

    # getting X & y values (extracted features, labels)
    X_train = X[train_mask] # taking all the X values, and keeping the true ones from train mask
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test, train_participants, test_participants


## --- SCALING DATA --- ##
from sklearn.preprocessing import StandardScaler

def scale_data(X_train, X_test):
    scaler = StandardScaler() # create object

    X_train_scaled = scaler.fit_transform(X_train) # fit & create parameters, then train X_train
    X_test_scaled = scaler.transform(X_test) # use prev parameters to train X_test

    return X_train_scaled, X_test_scaled, scaler


## --- BINARY TRAINING DATA --- ##
# parameters: X features, y features
def make_binary_data(X, y, participant_ids):
    # remove neutral 3
    mask = y != 3

    # mask everything
    X_binary = X[mask]
    y_filtered = y[mask]
    participant_ids_binary = participant_ids[mask]

    # make the y binary - 0 for low, 1 for high
    y_binary = (y_filtered >= 4).astype(int)

    return X_binary, y_binary, participant_ids_binary


## --- MAKE 3-CLASS DATA --- ##
def make_three_class_data(X, y, participant_ids):

    # make the three classes:
    # 1, 2 → 0 (low)
    # 3    → 1 (neutral)
    # 4, 5 → 2 (high)
    y_three_class = np.where(
        y <= 2, 0,
        np.where(y == 3, 1, 2)
    )

    return X, y_three_class, participant_ids

## --- FITTING AND TRAINING DATA FROM MODEL --- ##
# parameters: model is model of choice - need to define params of that beforehand
#             rest is just data from training & testing separation
def train_and_test_model(model, X_train, y_train, X_test, y_test):
    # train model
    model.fit(X_train, y_train)

    # predict training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # calculate accuracy
    train_accuracy = np.mean(y_train_pred == y_train)
    test_accuracy = np.mean(y_test_pred == y_test)

    print("Training accuracy:", train_accuracy)
    print("Testing accuracy:", test_accuracy)

    return model, y_train_pred, y_test_pred


## --- EMOTION SCORE COUNTS --- ##
def print_label_counts(y, label):
    print("Label Counts for " + label)
    for label in range(1, 6):
        print(
            label,
            ":",
            np.sum(y == label)
        )


## --- PREDICTION ACCURACY BY EMOTION SCORE --- ##
from sklearn.metrics import accuracy_score, recall_score

# parameters: y_test = the test correct results, y_pred = the predicted results
def pred_label_accuracy(y_test, y_pred):
    print("Percentage Correct by Score")
    for valence_score in range (1, 6):
        # make masks
        y_test_mask = np.isin(y_test, valence_score)
        correct = (y_test_mask) & (y_pred == valence_score)

        # count
        y_test_sum = np.sum(y_test_mask)
        correct_sum = np.sum(correct)

        # calculate
        print(valence_score, ": ", (correct_sum / y_test_sum)*100, "%")

    # overall percentage correct
    print("Overall accuracy:", np.mean(y_pred == y_test)*100, "%")


## --- DUMMY CLASSIFIER --- ##
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

# the dummy training part + accuracy
def dummy_baseline(X_train, y_train, X_test, y_test):
    dummy = DummyClassifier(strategy="most_frequent")

    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)

    dummy_accuracy = accuracy_score(y_test, dummy_pred)

    print("Dummy testing accuracy:", dummy_accuracy)

    return dummy, dummy_pred, dummy_accuracy

# the classification report
def print_classification_report(y_test, y_pred):
    print(classification_report(y_test, y_pred))