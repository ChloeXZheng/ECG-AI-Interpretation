import numpy as np 
# Date: 9/3/2026
# Goal: Get all the features into numbers in one row, so we can actually do ML
# [Mean, StDev, Theta power, Alpha power, Beta power, Gamma power]
# Mean = avg signal level, stdv = how much it fluctuates
# STRENGTH of: theta = 4-8 ; alpha = 8-13 ; beta = 13-30 ; gamma = 30-45
# aka how much of each in the window
# 6 features x 14 channels = 84 numbers per window

# flow:
# extract those features
# do it 13 more times for that sample
# put it all into a row

from scipy.signal import welch

# Function: extracts the 6 features
# Parameters: channel_signal = the EEG window data
#             sampling_rate = frequency of measurement
#             welch_segment_seconds = using 2 second pieces instead of one
def extract(channel_signal, sampling_rate = 128, welch_segment_seconds = 2): 

    # create dictionaries for frequencies & later band_power
    FREQUENCY_BANDS = {

        # appear during light sleep and REM dreaming, right as you
        # drift off to sleep/wake up
        "theta": (4, 8),

        # relaxed, wakeful states with closed eyes
        "alpha": (8, 13),

        # awake, alert, and focused on active tasks
        "beta": (13, 30),

        # deep focus, high-level problem solving, moments of sudden insight
        "gamma": (30, 45),
    }

    band_power = {}

    # uses a function welch to calculate how much of each freq there is
    frequencies, power = welch(
        channel_signal,
        sampling_rate,
        nperseg = sampling_rate * welch_segment_seconds
    )

    # looping through dictionary to sum up all the frequencies for each type of signal
    for band_name, (low_freq, high_freq) in FREQUENCY_BANDS.items():
        places_in_band = (frequencies >= low_freq) & (frequencies < high_freq)
        band_power[band_name] = power[places_in_band].sum()

    # getting the mean & stdev
    mean = channel_signal.mean()
    stdev = channel_signal.std()

    return band_power, mean, stdev


# Date: 9/3/2026
# Function: take one 4-second EEG window and turn all 14 channels into a single row
# of 84 numerical features that a machine-learning model can use
# Parameters: window = 4-second EEG window
# sampling_rate = frequency of measurement

def extract_window_features(window, sampling_rate = 128):

    features = []

    # establishes a range of 14 (because window.shape = (512, 14))
    # 512 samples every 4 seconds
    for channel in range (window.shape[1]):

        channel_signal = window[:, channel]

        band_power, mean, stdev = extract(channel_signal, sampling_rate = sampling_rate)

        # puts all 6 features into the list
        # repeats for each of the 14 EEG channels
        features.extend([
            mean, 
            stdev,
            band_power["theta"],
            band_power["alpha"],
            band_power["beta"],
            band_power["gamma"]
        ])

    return np.array(features, dtype = np.float32)



# Date: 9/4/2026
# Function: go through every EEG window in X and turn each window into its 84 numerical features 
# (all 21298 windows of 4-second EEG data)
# Param: X = the EEG data; sampling_rate = frequency of measurement

def extract_dataset_features(X, sampling_rate = 128):

    # creating an empty list where we'll eventually put 84 features from each of the 21298 windows
    all_features = []

    for window in X:
        features = extract_window_features(window, sampling_rate = sampling_rate)
        all_features.append(features)


    return np.array(all_features, dtype = np.float32)



# Date: 9/4/2026
# Function: save the feature matrix, so you don't have to run it over and over

def save_features(dataset_features, y, participant_ids, trial_ids, window_ids, file_path):
    np.savez_compressed(
        file_path,
        dataset_features = dataset_features,
        y=y, 
        participant_ids = participant_ids, 
        trial_ids = trial_ids, 
        window_ids = window_ids
    )








