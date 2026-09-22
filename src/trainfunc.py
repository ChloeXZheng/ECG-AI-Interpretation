# date: 9/16/2026
# Function: to make actual functions for the training so its not... really messy

import numpy as np

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.dummy import DummyClassifier
from sklearn.base import clone

## --- LOAD PROCESSED DATA --- ##
# parameter: file_path = "data/processed_features.npz"
def load_processed_data(file_path):
    data = np.load(file_path)

    X = data["dataset_features"]
    y = data["y"]
    participant_ids = data["participant_ids"]

    return X, y, participant_ids

## --- SPLITTING PARTICIPANTS INTO TEST & TRAIN --- ##
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
def scale_data(X_train, X_test):
    scaler = StandardScaler() # create object

    X_train_scaled = scaler.fit_transform(X_train) # fit & create parameters, then train X_train
    X_test_scaled = scaler.transform(X_test) # use parameters learned from X_train

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

    # training accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print("Training accuracy:", train_accuracy)

    # evaluate testing predictions
    results = evaluate_model(y_test, y_test_pred)

    return model, y_train_pred, y_test_pred, results


## --- EVALUATE MODEL --- ##
# Function: Evaluate model predictions.
# Parameters: y_test: actual labels
#            y_pred: predicted labels
# Returns: dictionary containing evaluation metrics
def evaluate_model(y_test, y_pred):

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    print("Testing accuracy:", accuracy)
    print("Macro F1:", macro_f1)
    print("Balanced accuracy:", balanced_accuracy)

    print("\nConfusion Matrix:")
    print(confusion)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": confusion
    }


## --- DUMMY CLASSIFIER --- ##
# the dummy training part + accuracy
def dummy_baseline(X_train, y_train, X_test, y_test):
    dummy = DummyClassifier(strategy="most_frequent")

    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)

    print("Dummy baseline:")
    results = evaluate_model(y_test, dummy_pred)

    return dummy, dummy_pred, results


## --- PARTICIPANT-LEVEL CROSS-VALIDATION --- ##
def evaluate_stratified_group_cv(model, X, y, participant_ids, n_splits=5, random_state=42):

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups=participant_ids),
        start=1
    ):

        # split data using participant indices
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # make a fresh copy of the model for this fold
        fold_model = clone(model)

        # train model
        fold_model.fit(X_train, y_train)

        # predict test data
        y_pred = fold_model.predict(X_test)

        # calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

        fold_results.append({
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_accuracy
        })

        print(f"Fold {fold}:")
        print("  Accuracy:", accuracy)
        print("  Macro F1:", macro_f1)
        print("  Balanced accuracy:", balanced_accuracy)
        print()

    # calculate mean and standard deviation across folds
    mean_accuracy = np.mean(
        [result["accuracy"] for result in fold_results]
    )
    std_accuracy = np.std(
        [result["accuracy"] for result in fold_results]
    )

    mean_macro_f1 = np.mean(
        [result["macro_f1"] for result in fold_results]
    )
    std_macro_f1 = np.std(
        [result["macro_f1"] for result in fold_results]
    )

    mean_balanced_accuracy = np.mean(
        [result["balanced_accuracy"] for result in fold_results]
    )
    std_balanced_accuracy = np.std(
        [result["balanced_accuracy"] for result in fold_results]
    )

    print("Cross-validation results:")
    print(f"Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Macro F1: {mean_macro_f1:.4f} ± {std_macro_f1:.4f}")
    print(
        f"Balanced accuracy: "
        f"{mean_balanced_accuracy:.4f} ± {std_balanced_accuracy:.4f}"
    )

    return {
        "fold_results": fold_results,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "mean_macro_f1": mean_macro_f1,
        "std_macro_f1": std_macro_f1,
        "mean_balanced_accuracy": mean_balanced_accuracy,
        "std_balanced_accuracy": std_balanced_accuracy
    }


"""
Archived code:

# the classification report
def print_dummy_classification_report(y_test, y_pred):
    print(classification_report(y_test, y_pred))

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
"""