# Date: 9/18/26
# Function: attempting to split data into 3 classes: 1-2 = 0, 3 = 1, 4-5 = 2

import src.trainfunc as trainfunc
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- LOAD PROCESSED DATA --- #
X, y, participant_ids = trainfunc.load_processed_data("data/processed_features.npz")

# Keep the original 1-5 labels
y_original = y.copy()

# --- CONVERT TO 3 CLASSES --- #
X, y, participant_ids = trainfunc.make_three_class_data(X, y, participant_ids)

# --- SPLIT DATA INTO TRAIN & TEST --- #
X_train, X_test, y_train, y_test, train_participants, test_participants = trainfunc.split_by_participant(X, y, participant_ids)

test_mask = np.isin(participant_ids, test_participants)
y_original_test = y_original[test_mask]

# --- INITIALIZE MODEL -- #
model = RandomForestClassifier(
    n_estimators = 500,
    min_samples_leaf = 20,
    random_state = 42, 
    n_jobs = -1 # means use all available CPU cores
)

# --- TRAIN & TEST MODEL --- #
model, y_train_pred, y_test_pred = trainfunc.train_and_test_model(
    model, 
    X_train,
    y_train,
    X_test,
    y_test
)

# --- FEATURE IMPORTANCE --- #

feature_importances = model.feature_importances_

# Reshape 84 features into 14 channels × 6 features
feature_importances = feature_importances.reshape(14, 6)

# Add the importance across all 14 channels
total_importance = feature_importances.sum(axis=0)

feature_names = [
    "Mean",
    "StDev",
    "Theta",
    "Alpha",
    "Beta",
    "Gamma"
]

print("\nFeature Importance:")

for name, importance in zip(feature_names, total_importance):
    print(name, ":", importance)




print("\nActual vs Predicted:")

# Actual counts
print("Actual Low (1/2):", np.sum(y_test == 0))
print("Actual Neutral (3):", np.sum(y_test == 1))
print("Actual High (4/5):", np.sum(y_test == 2))

# Predicted counts
print("Predicted Low (1/2):", np.sum(y_test_pred == 0))
print("Predicted Neutral (3):", np.sum(y_test_pred == 1))
print("Predicted High (4/5):", np.sum(y_test_pred == 2))



print("\nActual → Predicted:")

print("Low → Low:", np.sum((y_test == 0) & (y_test_pred == 0)))
print("Low → Neutral:", np.sum((y_test == 0) & (y_test_pred == 1)))
print("Low → High:", np.sum((y_test == 0) & (y_test_pred == 2)))

print("Neutral → Low:", np.sum((y_test == 1) & (y_test_pred == 0)))
print("Neutral → Neutral:", np.sum((y_test == 1) & (y_test_pred == 1)))
print("Neutral → High:", np.sum((y_test == 1) & (y_test_pred == 2)))

print("High → Low:", np.sum((y_test == 2) & (y_test_pred == 0)))
print("High → Neutral:", np.sum((y_test == 2) & (y_test_pred == 1)))
print("High → High:", np.sum((y_test == 2) & (y_test_pred == 2)))


print("\nOriginal Score → Predicted Group")

for score in range(1, 6):
    mask = y_original_test == score

    low = np.sum(y_test_pred[mask] == 0)
    neutral = np.sum(y_test_pred[mask] == 1)
    high = np.sum(y_test_pred[mask] == 2)

    print(
        score,
        "→ Low:", low,
        "Neutral:", neutral,
        "High:", high
    )