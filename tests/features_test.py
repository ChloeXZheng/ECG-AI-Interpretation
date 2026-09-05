import numpy as np
from src.load_data import load_dreamer
from src.dataset import build_dataset
from src.features import extract_window_features, extract_dataset_features, save_features

dreamer = load_dreamer("DREAMER.mat")

X, y, participant_ids, trial_ids, window_ids = build_dataset(dreamer)

window = X[0]

window_features = extract_window_features(window)

print("Original window shape:", window.shape)
print("Features shape:", window_features.shape)
print("Features dtype:", window_features.dtype)
print("Features:", window_features)

dataset_features = extract_dataset_features(X)

print("Dataset features shape:", dataset_features.shape)
print("Dataset features dtype:", dataset_features.dtype)


save_features(
    dataset_features,
    y,
    participant_ids,
    trial_ids,
    window_ids,
    "data/processed_features.npz"
)

print("FEATURES SAVED!")

