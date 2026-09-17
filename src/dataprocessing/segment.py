import numpy as np
# Date: 8/26/2026
# Purpose: Take a continuous EEG signal and break it into smaller windows of data

# creates a function called create_windows (EEG data -> create_windows -> windows)
# window_seconds=4 means that each eeg window = 4 seconds long
def create_windows (eeg, sampling_rate=128, window_seconds=4):

    # calculuates how many EEG samples should be in each window
    # 128 samples/second * 4 seconds = 512 samples per window = window_size
    window_size = sampling_rate * window_seconds

    num_windows = len(eeg) // window_size # num of complete windows = num of total samples / num of samples in one window

    # takes EEG from beginning up to, & not including, index of last sample that forms last possible window
    # gets rid of the excess that can't form a complete window
    trimmed_eeg = eeg[:num_windows * window_size] 

    # splits the long EEG recording of samples, into windows (reorganizes within NumPy)
    windows = trimmed_eeg.reshape(
        num_windows,
        window_size,
        eeg.shape[1]
    )

    # convert into float32 - half the size of current float64 (if it is a 64)
    windows = windows.astype(np.float32, copy=False) 

    # returns 3d numpy array
    return windows

"""
TESTING
eeg = np.random.rand(25472, 14)

windows = create_windows(eeg)

print("Number of windows:", len(windows))
print("Shape of first window:", windows[0].shape)
"""