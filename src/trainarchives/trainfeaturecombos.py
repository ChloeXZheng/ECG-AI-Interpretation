import numpy as np

# Date: 9/18/2026
# Experiment: Test combinations of EEG feature groups

import src.trainfunc as trainfunc
from sklearn.ensemble import RandomForestClassifier


# --- LOAD PROCESSED DATA --- #

X, y, participant_ids = trainfunc.load_processed_data(
    "data/processed_features.npz"
)


# --- CONVERT TO 3 CLASSES --- #

# 1, 2 -> Low
# 3    -> Neutral
# 4, 5 -> High

X, y, participant_ids = trainfunc.make_three_class_data(
    X, y, participant_ids
)


# --- SPLIT INTO TRAIN & TEST --- #

X_train, X_test, y_train, y_test, train_participants, test_participants = trainfunc.split_by_participant(
    X, y, participant_ids
)


# --- ORGANIZE FEATURES --- #

# Each window has:
# 14 channels × 6 features
#
# Feature order:
# 0 = Mean
# 1 = StDev
# 2 = Theta
# 3 = Alpha
# 4 = Beta
# 5 = Gamma

X_train_organized = X_train.reshape(-1, 14, 6)
X_test_organized = X_test.reshape(-1, 14, 6)


# --- CREATE FEATURE GROUPS --- #

mean_train = X_train_organized[:, :, 0]
mean_test = X_test_organized[:, :, 0]

stdev_train = X_train_organized[:, :, 1]
stdev_test = X_test_organized[:, :, 1]

frequency_train = X_train_organized[:, :, 2:6].reshape(-1, 56)
frequency_test = X_test_organized[:, :, 2:6].reshape(-1, 56)


# --- TEST FEATURE COMBINATIONS --- #

combinations = {
    "Mean + StDev": (
        mean_train,
        mean_test,
        stdev_train,
        stdev_test
    ),

    "Mean + Frequency": (
        mean_train,
        mean_test,
        frequency_train,
        frequency_test
    ),

    "StDev + Frequency": (
        stdev_train,
        stdev_test,
        frequency_train,
        frequency_test
    )
}


for name, (train_a, test_a, train_b, test_b) in combinations.items():

    # Combine the two feature groups side-by-side
    X_train_combo = np.concatenate(
        [train_a, train_b],
        axis=1
    )

    X_test_combo = np.concatenate(
        [test_a, test_b],
        axis=1
    )

    print("\n---", name, "---")
    print("Feature shape:", X_train_combo.shape)

    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    model, y_train_pred, y_test_pred = trainfunc.train_and_test_model(
        model,
        X_train_combo,
        y_train,
        X_test_combo,
        y_test
    )



print("DUMMYYYYYYYYYYYYY")
trainfunc.dummy_baseline(
    X_train,
    y_train,
    X_test,
    y_test
)
