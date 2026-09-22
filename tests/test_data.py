# date: 9/21/2026
# function: tests data loading and label transformations


import numpy as np
import sys

sys.path.append("src")

import trainfunc as tf


## --- LOAD DATA --- ##

X, y, participant_ids = tf.load_processed_data(
    "data/processed_features.npz"
)


## --- BASIC DATA TESTS --- ##

assert len(X) == len(y), (
    "X and y have different numbers of samples"
)

assert len(X) == len(participant_ids), (
    "X and participant_ids have different numbers of samples"
)

assert X.ndim == 2, (
    "X should be a 2D array"
)

assert X.shape[1] == 84, (
    f"Expected 84 features, got {X.shape[1]}"
)

assert not np.isnan(X).any(), (
    "X contains NaN values"
)


## --- PRINT DATA INFO --- ##

print("Original data:")
print("  X shape:", X.shape)
print("  Number of labels:", len(y))
print("  Number of participants:", len(np.unique(participant_ids)))
print("  Labels:", np.unique(y))


## --- BINARY DATA --- ##

X_binary, y_binary, participants_binary = tf.make_binary_data(
    X,
    y,
    participant_ids
)

assert len(X_binary) == len(y_binary)
assert len(X_binary) == len(participants_binary)

assert set(np.unique(y_binary)) == {0, 1}, (
    "Binary labels should only contain 0 and 1"
)


## --- THREE CLASS DATA --- ##

X_three, y_three, participants_three = tf.make_three_class_data(
    X,
    y,
    participant_ids
)

assert len(X_three) == len(y_three)
assert len(X_three) == len(participants_three)

assert set(np.unique(y_three)) == {0, 1, 2}, (
    "Three-class labels should only contain 0, 1, and 2"
)


## --- FINALS --- ##

print()
print("Binary data:")
print("  X shape:", X_binary.shape)
print("  Labels:", np.unique(y_binary))

print()
print("Three-class data:")
print("  X shape:", X_three.shape)
print("  Labels:", np.unique(y_three))

print()
print("All data tests passed!")