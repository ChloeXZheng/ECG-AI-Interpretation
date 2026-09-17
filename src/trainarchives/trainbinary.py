import numpy as np
from sklearn.model_selection import train_test_split

# Date: 9/16/2026
# Function: Trying out a binary training thing



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


"""
## --- ASSESSING PARTICIPANT DATA --- ###
from collections import Counter

print("Number of participants:", len(unique_participants))

for participant in unique_participants:
    participant_labels = y[participant_ids == participant]
    binary_labels = participant_labels[participant_labels != 3]

    print(
        participant,
        "total:", len(participant_labels),
        "low:", np.sum(binary_labels <= 2),
        "high:", np.sum(binary_labels >= 4)
    )
"""

## --- MASKING TO TEST TRAINING MODEL --- ##
# partition the unique participants into training and testing sets
train_participants, test_participants = train_test_split(
    unique_participants,
    test_size = 0.2, # 20% of participants will be used for testing (aka only 5 ppl)
    random_state = 42 # fixed starting point for random selection; every time get same participants in training/testing
)

# mask = an array of True and False values that acts like a filter
# assigns true/false to each of 21,298 windows. true = in training set, false = in testing set
train_mask = np.isin(participant_ids, train_participants) 

# creates an equivalent mask for test participants. true = in testing set, false = in training set
test_mask = np.isin(participant_ids, test_participants)

# getting X & y values (extracted features, labels)
X_train = X[train_mask] # taking all the X values, and keeping the true ones from train mask
X_test = X[test_mask] 
y_train = y[train_mask] 
y_test = y[test_mask] 



## --- MAKING IT BINARY --- ##
# remove neutral valence (3)
train_mask_binary = y_train != 3
test_mask_binary = y_test != 3

# getting the X & y values
X_train_binary = X_train[train_mask_binary]
X_test_binary = X_test[test_mask_binary]
y_train_binary = (y_train[train_mask_binary] >= 4).astype(int)
y_test_binary = (y_test[test_mask_binary] >= 4).astype(int)



## --- TRAINING --- ##
from sklearn.ensemble import RandomForestClassifier

# initialize model
binary_model = RandomForestClassifier(
    n_estimators=500,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)

# fit the model with training data
binary_model.fit(X_train_binary, y_train_binary)

# test the model
binary_pred = binary_model.predict(X_test_binary)

# print accuracies
print("Binary training accuracy:",
      binary_model.score(X_train_binary, y_train_binary))

print("Binary testing accuracy:",
      binary_model.score(X_test_binary, y_test_binary))

# Results:
# Binary training accuracy: 0.918488160291439
# Binary testing accuracy: 0.5839714139867279

# Analysis: 
# the 58% is more than around 50% baseline, which means its not just completely guessing...
# but hugeee gap between training and testing - "generalization problem"


"""
## --- ESTABLISH BASELINE - DUMMY --- ##
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

# dummy accuracy = what u get by predicting whichever class is more common
dummy = DummyClassifier(strategy="most_frequent")

# fit the model
dummy.fit(X_train_binary, y_train_binary)

# test the model
dummy_pred = dummy.predict(X_test_binary)

# print accuracies & stuff
print("Dummy testing accuracy:",
      accuracy_score(y_test_binary, dummy_pred))

print("\nRandom Forest classification report:")
print(classification_report(y_test_binary, binary_pred))

Results:
Dummy testing accuracy: 0.5387953037263911 (RF only 4.5% better than dummy...)

Random Forest classification report:
              precision    recall  f1-score   support

           0       0.54      0.72      0.61      1807
           1       0.66      0.47      0.55      2111

    accuracy                           0.58      3918
   macro avg       0.60      0.59      0.58      3918
weighted avg       0.60      0.58      0.58      3918
"""
