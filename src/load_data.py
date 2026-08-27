# Date: 8/25/2026
# Purpose: Contains method for loading dataset DREAMER.mat

# importing function loadmat() from module io in library scipy
from scipy.io import loadmat 

# function: loads the matlab file into the project
# parameter: path (location) of desired file
def load_dreamer(file_path):

    # calls loadmat() function, stores results in variable 'dreamer'
    dreamer = loadmat(
        file_path, # giving function the path/file that it shld open
        squeeze_me=True, # reduces unnecessary dimensions when importing .mat file into python
        struct_as_record=False # allows for more convenient access of fields (e.g. age)
    )

    # returns only the dataset
    # (because dreamer contains both metadata & the actual data like mat file version)
    return dreamer["DREAMER"]