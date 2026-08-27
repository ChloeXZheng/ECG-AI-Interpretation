# Date: 8/26/2026
# Purpose: Take a continuous EEG signal and break it into smaller windows of data

def create_windows(eeg, sampling rate=128, window_seconds=4):
    # creates a function called create_windows (EEG data -> create_windows -> windows)
    # window_seconds=4 means that each eeg window = 4 seconds long

    window_size = sampling_rate * window_seconds
    # calculuates how many EEG samples should be in each window
    # 128 samples/second * 4 seconds = 512 samples per window

    windows = []
    # creates an empty list of windows, will put each eeg window into this list

    for start in range(0, len(eeg) - window_size+1, window_size):
        # python method: range(a,b,c) = creates a range object starting at a, ending at b-1, incrementing by c
        # stopping position = len(eeg)-window_size+1; 24961 range upper limit

        end = start + window_size
        # know when the window ends
        
        windows.append(eeg[start:end])
        # extract the window; tells it to give eeg data from index start to index end
        # random: python .append() method adds a single element to the very end of an existing list

    return windows
 
# TESTING
eeg = np.random.rand(25472, 14)

windows = create_windows(eeg)

print("Number of windows:", len(windows))
print("Shape of first window:", windows[0].shape)
