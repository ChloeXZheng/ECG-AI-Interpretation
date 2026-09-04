import numpy as np
from src.load_data import load_dreamer
from src.dataset import build_dataset
from src.features import extract_window_features

dreamer = load_dreamer("DREAMER.mat")

X, y, participant_ids, trial_ids, window_ids = build_dataset(dreamer)

window = X[0]

features = extract_window_features(window)

print("Original window shape:", window.shape)
print("Features shape:", features.shape)
print("Features dtype:", features.dtype)
print("Features:", features)