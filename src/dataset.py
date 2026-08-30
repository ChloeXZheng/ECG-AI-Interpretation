# Date: 8/27/2026
# Purpose: functions for breaking down trial/patient data + compiles it

import numpy as np

from .segment import create_windows

# Function: inputs 3 functions from preprocess.py to use inside dataset.py
from .preprocess import (

    # measures EEG relative to patient's starting baseline
    baseline_correct, 

    # creates and designs filter to remove noise from EEG data 
    create_bandpass_filter,

    # applies the filter to the data
    apply_bandpass_filter,
)


# Function: creates parallel arrays - data + label + metadata
# Parameters: eeg = the eeg data
#             label = the trial this is from
#             the three types of id = metadata from dataset
#             sampling rate = the sampling rate ; window_seconds = the length of a window
def add_trial_to_dataset(eeg, label, participant_id, trial_id, sampling_rate=128, window_seconds=4):

    # previous function from segment.py, creates the windows
    windows = create_windows(
        eeg,
        sampling_rate=sampling_rate,
        window_seconds=window_seconds
    )

    # recalculates the # of windows
    num_windows = len(eeg) // (sampling_rate * window_seconds)

    # EMOTION LABEL!
    labels = np.full(
        num_windows,
        label,
        dtype=np.uint8
    )

    # METADATA
    participant_ids = np.full(
        num_windows,
        participant_id,
        dtype=np.uint8
    )

    trial_ids = np.full(
        num_windows,
        trial_id,
        dtype=np.uint8
    )

    window_ids = np.arange(
        num_windows,
        dtype=np.uint16
    )

    # returns the separated windows alongside their labels
    return windows, labels, participant_ids, trial_ids, window_ids

# Function: Converts all of the compiled arrays into NumPy arrays 
# Parameters: they are all of the parallel arrays
def combine_trials(all_windows, all_labels, all_participant_ids, all_trial_ids, all_window_ids):
    # data + label
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)

    # metadata
    participant_ids = np.concatenate(all_participant_ids)
    trial_ids = np.concatenate(all_trial_ids)
    window_ids = np.concatenate(all_window_ids)

    # returns numpy arrays
    return X, y, participant_ids, trial_ids, window_ids


"""
NOTES:
np.full():
    np.full(shape, fill_value, dtype=None)
        shape: the size/dimensions of the array
        fill_value: the value placed in every position
        dtype: optional data type used to store the values
"""

# ACTUALLY BUILDING THE DATASET!!!

# Date: 8/29/2026
# Function: builds the dataset from the DREAMER data!

def build_dataset(dreamer):

    # create empty lists
    all_windows = []
    all_labels = []
    all_participant_ids = []
    all_trial_ids = []
    all_window_ids = []

    # creating filter
    b, a = create_bandpass_filter()

    # loop through every participant
    for participant_id, participant in enumerate (dreamer.Data):

        # determine number of trials
        for trial_id in range (len(participant.EEG.stimuli)):

            # get the EEG recorded during this trial/video
            stimulus = participant.EEG.stimuli[trial_id]

            # get the baseline EEG recorded before this trial/ivdeo
            baseline = participant.EEG.baseline[trial_id]

            # get this participant's emotional valence score for this trial
            label = participant.ScoreValence[trial_id]

            # correct baseline for the EEG
            baseline_corrected = baseline_correct(stimulus, baseline)

            # filter baseline-corrected EEG to 4-45 Hz (remove noise)
            filtered = apply_bandpass_filter(baseline_corrected, b, a)

            # add this trial to the dataset, calls function above
            windows, labels, participant_ids, trial_ids, window_ids = (
                add_trial_to_dataset (
                    filtered,
                    label,
                    participant_id,
                    trial_id
                )
            )

            # append this trials' data to the lists
            all_windows.append(windows)
            all_labels.append(labels)
            all_participant_ids.append(participant_ids)
            all_trial_ids.append(trial_ids)
            all_window_ids.append(window_ids)

    # combine trials from ALL participants into one final dataset
    X, y, participant_ids, trial_ids, window_ids = combine_trials(
        all_windows,
        all_labels,
        all_participant_ids,
        all_trial_ids,
        all_window_ids
    )

    # return the completed dataset
    return X, y, participant_ids, trial_ids, window_ids






