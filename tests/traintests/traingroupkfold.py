# Date: 9/16/2026
# Function: After testing the binary way of splitting participants, want to test it for whether the new accuracy is good
# comes after: trainbinary.py

import numpy as np
import trainfunc as tf

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score


## --- LOAD DATA + MAKE DATA BINARY --- ##
X, y, participant_ids = tf.load_processed_data("data/processed_features.npz")

X_binary, y_binary, groups_binary = tf.make_binary_data(X, y, participant_ids)


## --- 5-FOLD PARTICIPANT CROSS-VALIDATION + DUMMY --- ##
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_accuracies = []
dummy_accuracies = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X_binary, y_binary, groups=groups_binary),
    start=1
):

    X_train_binary = X_binary[train_idx]
    X_test_binary = X_binary[test_idx]

    y_train_binary = y_binary[train_idx]
    y_test_binary = y_binary[test_idx]

    train_participants = np.unique(groups_binary[train_idx])
    test_participants = np.unique(groups_binary[test_idx])

    print(f"\n--- Fold {fold} ---")
    print("Training participants:", train_participants)
    print("Testing participants:", test_participants)

    model = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_binary, y_train_binary)

    binary_pred = model.predict(X_test_binary)

    accuracy = accuracy_score(y_test_binary, binary_pred)
    fold_accuracies.append(accuracy)

    print("Testing accuracy:", accuracy)

    ## --- ESTABLISH BASELINE - DUMMY --- ##
    dummy, dummy_pred, dummy_accuracy = tf.dummy_baseline(X_train_binary, y_train_binary, X_test_binary, y_test_binary)

    dummy_accuracies.append(dummy_accuracy)



## --- OVERALL RESULT --- ##

print("\n--- RANDOM FOREST RESULTS ---")

print("Fold accuracies:")
print(fold_accuracies)

print("Mean accuracy:", np.mean(fold_accuracies))
print("Standard deviation:", np.std(fold_accuracies))


print("\n--- DUMMY RESULTS ---")

print("Fold accuracies:")
print(dummy_accuracies)

print("Mean accuracy:", np.mean(dummy_accuracies))
print("Standard deviation:", np.std(dummy_accuracies))