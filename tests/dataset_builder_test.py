# Date: 8/30/2026
# Function: testing the build_dataset func

import numpy as np
from src.load_data import load_dreamer
from src.dataset import build_dataset, save_dataset

# load the data
dreamer = load_dreamer('DREAMER.mat')

# build dataset
X, y, participant_ids, trial_ids, window_ids = build_dataset(dreamer)

save_dataset(X, y, participant_ids, trial_ids, window_ids, "data/processed_dataset.npz")
print("Dataset saved successfully!")

# print & test
print(X.shape)
print(y.shape)
print(participant_ids.shape)
print(trial_ids.shape)
print(window_ids.shape)

print(X.dtype)
print(np.unique(y, return_counts=True))
print("Memory in MB:", X.nbytes / 1024**2)
print("NaNs:", np.isnan(X).sum())

print("Infinite values:", np.isinf(X).sum())
print("Participants:", np.unique(participant_ids, return_counts=True))
print("Trials:", np.unique(trial_ids, return_counts=True))
print("Minimum EEG value:", X.min())
print("Maximum EEG value:", X.max())
print("Mean EEG value:", X.mean())
print("Standard deviation:", X.std())

# investigating the min & max eeg values 
print(np.percentile(X, [0, 0.1, 1, 50, 99, 99.9, 100]))

min_location = np.unravel_index(np.argmin(X), X.shape)
max_location = np.unravel_index(np.argmax(X), X.shape)

print("Minimum location:", min_location)
print("Maximum location:", max_location)