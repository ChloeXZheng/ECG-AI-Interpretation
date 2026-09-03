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
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
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

def do_it_14_times():
    
    return 