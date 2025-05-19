import numpy as np
import os
import pbdlib as pbd
import pickle
import sys
import time

# Training data location
## Windows path
# rel_path = "C:\\Users\\ven058\\OneDrive - Wageningen University & Research\\"
## Linux path
rel_path = "/media/data/DataRobert/"

# Required for rel import
if rel_path.startswith("C:"):
    new_path = "C:\\Users\\ven058\\orchard_code\\orchard-reactive-tp-gmr\\lfd_executor\\lfd_executor\\pbdlib_custom"
else:
    new_path = "/home/lfd/orchard_ws/src/orchard-reactive-tp-gmr/lfd_executor/lfd_executor/pbdlib_custom"

if new_path not in sys.path:
    sys.path.append(new_path)
from gmmr import GMMR

# Load data
approach_path = os.path.join(rel_path, "PhD", "05 Study 4", "04 Robot", "training_data", "approach")
retreat_path = os.path.join(rel_path, "PhD", "05 Study 4", "04 Robot", "training_data", "retreat")

filelist_approach = os.listdir(approach_path)
filelist_retreat = os.listdir(retreat_path)

data_approach = []
for f in filelist_approach:
    temp = np.genfromtxt(approach_path + '/' + f, delimiter=',')
    data_approach.append(temp)

data_retreat = []
for f in filelist_retreat:
    temp = np.genfromtxt(retreat_path + '/' + f, delimiter=',')
    data_retreat.append(temp)

# TODO: Just for testing, remove for final version
# Duplicate the data to increase the number of demonstrations to 40
# data_approach = data_approach * 20
# data_retreat = data_retreat * 20

# Model settings: 
dep_mask_split = np.zeros([13,13])
dep_mask_split[:7,:7] = 1
dep_mask_split[7:,7:] = 1
dep_mask_split[0,:] = 1
dep_mask_split[:,0] = 1

dep_masks = {
    "Split": dep_mask_split,
}

# Save models
store_path = os.path.join(rel_path, "PhD", "05 Study 4", "04 Robot", "models")
if not os.path.exists(store_path):
    os.makedirs(store_path)

for mask_key in dep_masks:
    for i in range(10):
        # Train GMM
        gmr_model_approach = GMMR(nb_states=5, num_iter=200, dep_mask=dep_masks[mask_key])
        approach_LL = gmr_model_approach.fit(data_approach)

        model_time = time.time()

        approach_model_name = "gmr_demos-{}_LL-{}_depMask-{}_trainTime-{}.pickle".format(
            len(data_approach), 
            approach_LL,
            mask_key,
            model_time)
        
        with open(os.path.join(store_path, "approach", approach_model_name), 'wb') as f:
            pickle.dump(gmr_model_approach, f)

        gmr_model_retreat = GMMR(nb_states=5, num_iter=200, dep_mask=dep_masks[mask_key])
        retreat_LL = gmr_model_retreat.fit(data_retreat)

        retreat_model_name = "gmr_demos-{}_LL-{}_depMask-{}_trainTime-{}.pickle".format(
            len(data_approach), 
            retreat_LL,
            mask_key,
            model_time)

        with open(os.path.join(store_path, "placing", retreat_model_name), 'wb') as f:
            pickle.dump(gmr_model_retreat, f)
