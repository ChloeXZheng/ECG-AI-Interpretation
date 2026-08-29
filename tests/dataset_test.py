# Date: 8/28/2026
# Function: test helper functions in dataset.py with single trial/participant b4 making bigger function

from src.load_data import load_dreamer
from src.preprocess import (
    baseline_correct, 
    create_bandpass_filter, 
    apply_bandpass_filter
)
from src.dataset import add_trial_to_dataset

# load the data
dreamer = load_dreamer('data/DREAMER.mat')

# access the participant & the trial 
participant_id = 0
trial_id = 0

# access participant data
participant = dreamer.Data[participant_id]

# access trial data
stimulus = participant.EEG.stimuli[trial_id]
baseline = participant.EEG.baseline[trial_id]

# access Valence score
label = participant.ScoreValence[trial_id]

# preprocess
baseline_corrected = baseline_correct(stimulus, baseline)
b, a = create_bandpass_filter()
filtered = apply_bandpass_filter(baseline_corrected, b, a)

# feed that into add trial to dataset, build the parallel arrays
windows, labels, participant_ids, trial_ids, window_ids = (
    add_trial_to_dataset(
        filtered,
        label,
        participant_id=participant_id,
        trial_id=trial_id
    )
)

# inspect
print("Windows:", windows.shape, windows.dtype)
print("Labels:", labels.shape, labels.dtype)
print("Participant IDs:", participant_ids.shape, participant_ids.dtype)
print("Trial IDs:", trial_ids.shape, trial_ids.dtype)
print("Window IDs:", window_ids.shape, window_ids.dtype)


