# Date: 9/18/2026
# Experiment: Test predictive value of frequency-band EEG features
#
# Uses only:
# Theta, Alpha, Beta, Gamma
# from each of the 14 EEG channels.

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


# --- KEEP ONLY FREQUENCY FEATURES --- #

# Each channel has 6 features:
# Mean, StDev, Theta, Alpha, Beta, Gamma
#
# 2:6 selects:
# Theta, Alpha, Beta, Gamma
# from every channel.

X_train_frequency = X_train.reshape(-1, 14, 6)[:, :, 2:6].reshape(-1, 56)
X_test_frequency = X_test.reshape(-1, 14, 6)[:, :, 2:6].reshape(-1, 56)

print("Original feature shape:", X_train.shape)
print("Frequency-only feature shape:", X_train_frequency.shape)


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
    X_train_frequency,
    y_train,
    X_test_frequency,
    y_test
)