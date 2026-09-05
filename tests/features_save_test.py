import numpy as np

data = np.load("data/processed_features.npz")

print("Keys:", data.files)

print("Features shape:", data["dataset_features"].shape)
print("Labels shape:", data["y"].shape)
print("Participant IDs shape:", data["participant_ids"].shape)
print("Trial IDs shape:", data["trial_ids"].shape)
print("Window IDs shape:", data["window_ids"].shape)

print("Features dtype:", data["dataset_features"].dtype)
print("Labels dtype:", data["y"].dtype)