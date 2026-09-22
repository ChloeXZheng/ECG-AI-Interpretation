# Date: 9/18/2026
# Experiment: Test predictive value of Mean EEG features
#
# Uses only the Mean feature from each of the 14 EEG channels.
# Original feature structure:
# Mean, StDev, Theta, Alpha, Beta, Gamma
# repeated for 14 channels.

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


# --- KEEP ONLY MEAN FEATURES --- #

# Each channel has 6 features:
# Mean, StDev, Theta, Alpha, Beta, Gamma
#
# 0::6 selects:
# Mean from channel 0
# Mean from channel 1
# ...
# Mean from channel 13

X_train_mean = X_train[:, 0::6]
X_test_mean = X_test[:, 0::6]

print("Original feature shape:", X_train.shape)
print("Mean-only feature shape:", X_train_mean.shape)


# --- INITIALIZE MODEL --- #

model = RandomForestClassifier(
    n_estimators=500,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)


# --- TRAIN & TEST --- #

model, y_train_pred, y_test_pred = trainfunc.train_and_test_model(
    model,
    X_train_mean,
    y_train,
    X_test_mean,
    y_test
)