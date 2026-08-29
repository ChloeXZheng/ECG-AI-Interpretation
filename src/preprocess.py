# Date: 8/25/2026
# Purpose: Methods for processing raw data

# imports the library NumPy, and gives it the nickname "np" for "baseline_correct()" method
import numpy as np

# Function: for a SINGLE video/trial, corrects stimulus based on average of baseline for each electrode
# Parameters: stimulus = EEG recorded during the video
#             baseline = EEG recorded BEFORE that same video/trial
def baseline_correct(stimulus, baseline):
    # collapse the axis 0 (the 7808 samples) by taking its mean for baseline
    # aka taking the mean of all the samples, for each of the 14 electrodes
    baseline_mean = np.mean(baseline, axis=0) 

    # shows how the eeg value is changing, in relation to the mean
    # so that the numbers aren't all in the 4000's
    corrected = stimulus - baseline_mean

    # returns the corrected values
    return corrected


"""
elaboration on the baseline_mean:
    - basically, our data looks like:
        - axis 0: all the samples we have OVER TIME, for each electrode (think: y-axis, but not quite)
        - axis 1: all the 14 electrodes (think: x-axis, but not quite)
    -> by doing "np.mean()", we average the samples for each electrode (MAIN POINT!!!!) <-

    - some extra context:
        - sampling rate = 128 hz 
        - 128 hz = 128 rows per sample, for every second of EEG recording
        - so we might have like 7808 samples for one person/video/trial, which is a lot! and needs to be averaged
"""

# imports butter, filtfilt methods from module signal in library scipy, for "bandpass_filter()" method
from scipy.signal import butter, filtfilt

# Function: filters out the frequencies we don't want. SEE BELOW FUNCTIONS
# Parameters: eeg = the eeg data we wanna process
#             sampling_rate = how many times per sec EEG is measured ; default value can be overridden
#             low = the lowest frequency we want to keep ; default value can be overridden
#             high = the highest frequency we want to keep ; default value can be overridden
def bandpass_filter(eeg, sampling_rate=128, low=4, high=45): # not harcoding these default values = reusable function
    # designs the filter (Butterworth filter) for a gradual transition @ the cutoff points (4, 45)
    # b & a basically describe our filter mathematically
    b, a = butter(
        4, # 4th order Butterworth filter - controls how sharp the transition/cutoff is (higher = sharper)
        [low, high], # the low & high cutoffs ; scipy will adjust this relative to Nyquist frequency 
        btype="band", # type of filter (band basically means remove then keep then remove, other ones r like, remove then keep, or keep then remove)
        fs= sampling_rate # the frequecy of sampling
    )

    filtered = filtfilt(
        b, a, # the formula from the butter() method
        eeg, # data we're filtering (should be corrected alr)
        axis=0 # filter through time
    )

    return filtered

## ORRR we can create two separate functions
# this means we create the filter FIRST so we dont have to recalculate all the time
def create_bandpass_filter(sampling_rate=128, low=4, high=45):
    b, a = butter(
        4,
        [low, high],
        btype="band",
        fs=sampling_rate
    )

    return b, a

# and this is actually doing the filtering
def apply_bandpass_filter(eeg, b, a):
    return filtfilt(
        b,
        a,
        eeg,
        axis=0
    )

"""
elaboration on the cutoff numbers for frequency:
    - because sampling rate = 128...
    - Nyquist frequency: 128 / 2 = 64 Hz (data can only represent frequencies below 64 hz)

theta: 4-8 Hz
alpha: 8-13 Hz
beta: 13-30 Hz
gamma: 30-45 Hz

less than 4 = focusing on EEG frequency bands used for emotional recognition
more than 45 = can come from things like muscle activity and electrical noise

based on past work from: https://www.mdpi.com/2076-3425/16/7/716
"""