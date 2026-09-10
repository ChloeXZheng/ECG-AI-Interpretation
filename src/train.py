import numpy as np
from sklearn.model_selection import train_test_split

# Date: 9/8/2026
# Function: Actual training!

## --- EXTRACT DATA, SPLITTING BY PARTICIPANTS ---  ##
# load saved data into train.py
data = np.load("data/processed_features.npz")

# pull out feature matrix data
X = data["dataset_features"] # now X isn't window, like it was in prev file. 

# pull out valence labels for each window
y = data["y"]

# pull out participant ids, eventually split data so diffferent participants used for training and testing
participant_ids = data["participant_ids"]

# save array of unique participants
unique_participants = np.unique(participant_ids)

## --- MASKING TO TEST TRAINING MODEL --- ##
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

# getting X & y values (extracted features, labels)
X_train = X[train_mask] # taking all the X values, and keeping the true ones from train mask
X_test = X[test_mask] 
y_train = y[train_mask] 
y_test = y[test_mask] 

"""
print("X shape:", X.shape) # (21298, 84)
print("X_train shape:", X_train.shape) # (16668, 84)
print("X_test shape:", X_test.shape) # (4630, 84)
print("y_train shape:", y_train.shape) # (16668,)
print("y_test shape:", y_test.shape) # (4630,)
"""

## --- SCALING DATA --- ##
# not using test data to scale bc then that'd leak the test data. only using train, then applying scaling parameters to test
# scaling because of really low accuracy in training. but scaling didn't change anything...
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() # create an object

X_train_scaled = scaler.fit_transform(X_train) # fit & create parameters,then train X_train
X_test_scaled = scaler.transform(X_test) # use prev parameters to train X_test

print(np.mean(X_train_scaled, axis=0))
print(np.std(X_train_scaled, axis=0))


## --- TRAINING --- ##

from sklearn.linear_model import LogisticRegression

# initialize model
model = LogisticRegression(solver = 'newton-cholesky', max_iter=1000) 
# max_iter 100: [8.44%, 12.31%, 51.12%, 20.31%, 21.91%]
# max_iter 1000: [4.61%, 0.13%, 66.57%, 43.57%, 3.71%]

# train it
model.fit(X_train, y_train)

# test it
y_pred = model.predict(X_test)

# assess the test
percent_correct = []
for valence_score in range (1, 6):
    # make masks
    y_test_mask = np.isin(y_test, valence_score)
    correct = (y_test_mask) & (y_pred == valence_score)

    # count
    y_test_sum = np.sum(y_test_mask)
    correct_sum = np.sum(correct)

    # calculate
    percent_correct.append(correct_sum / y_test_sum)

print(percent_correct)
