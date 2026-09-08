import numpy as np
from sklearn.model_selection import train_test_split

# Date: 9/8/2026
# Function: Actually training the model!!!

# load saved data into train.py
data = np.load("data/processed_features.npz")

# pull out feature matrix data
X = data["dataset_features"]

# pull out valence labels for each window
y = data["y"]

# pull out participant ids, eventually split data so diffferent participants used for training and testing
participant_ids = data["participant_ids"]

# save array of unique participants
unique_participants = np.unique(participant_ids)

# partition the unique participants into training and testing sets
train_participants, test_participants = train_test_split(
    unique_participants,
    test_size = 0.2, # 20% of participants will be used for testing
    random_state = 42 # fixed starting point for random selection; every time get same participants in training/testing
)

# mask = an array of True and False values that acts like a filter
# assigns true/false to each of 21,298 windows. true = in training set, false = in testing set
train_mask = np.isin(participant_ids, train_participants)

# creates an equivalent mask for test participants. true = in testing set, false = in training set
test_mask = np.isin(participant_ids, test_participants)

# after testing, Training windows: 16668, Testing windows: 4630

