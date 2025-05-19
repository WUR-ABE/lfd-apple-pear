import numpy as np
import os
import pbdlib as pbd
import pickle
import sys

# Required for rel import
new_path = "/home/agrolegion/thesis_ws/src/includes/pbdlib_custom"

if new_path not in sys.path:
    sys.path.append(new_path)
from gmmr import GMMR
from hmmlqr import HMMLQR

dp_1 = os.path.dirname(pbd.__file__) + '/data/demos/demo_apple/corrected_approach'
filelist = next(os.walk(dp_1))[2]
data_1 = []
for f in filelist:
    temp = np.genfromtxt(dp_1 + '/' + f, delimiter=',')
    data_1.append(temp)

select_1 = range(0, len(data_1[0]), 20)
data_short_1 = []
for d in data_1:
    data_short_1.append(d[select_1, :])

dp_2 = os.path.dirname(pbd.__file__) + '/data/demos/demo_apple/corrected_placing'
filelist = next(os.walk(dp_2))[2]
data_2 = []
for f in filelist:
    temp = np.genfromtxt(dp_2 + '/' + f, delimiter=',')
    data_2.append(temp)

select_2 = range(0, len(data_2[0]), 20)
data_short_2 = []
for d in data_2:
    data_short_2.append(d[select_2, :])

full_data_train = [data_short_1, data_short_2]
# data_split_train, data_split_test = sk.model_selection.train_test_split(data_short, test_size=0.2, random_state=1337)

# Set storage path
store_path = "/home/agrolegion/thesis_ws/src/includes/pbdlib_custom/models/apple_harvesting/"

parts = ["approach", "placing"]
params_states = [4, 3, 2, 1]
params_hmm_u = [-4.0, -2.0, 0.0, 2.0, 4.0]
params_demo_count = [30, 50, 70, 90]

for part_num, part in enumerate(parts):
    data_split_train = full_data_train[part_num]
    for count in params_demo_count:
        print("Training model part %s, param count at %i" % (part, count))
        # Get right amount of demos
        data_train = data_split_train[0:count]

        # Train GMM
        gmr_model = GMMR()
        gmr_model.fit(data_train)

        # Train HMM
        lqr_model = HMMLQR()
        lqr_model.fit(data_train)

        # Pickling the results
        gmr_string = store_path + part + "/gmr/" + 'demos_' + str(count) + '.pickle'
        pickle.dump(gmr_model, open(gmr_string, 'wb'))

        lqr_string = store_path + part + "/lqr/" + 'demos_' + str(count) + '.pickle'
        pickle.dump(lqr_model, open(lqr_string, 'wb'))

    data_train = data_split_train
    for states in params_states:
        print("Training model part %s, param states at %i" % (part, states))
        # Train GMM
        gmr_model = GMMR(nb_states=states)
        gmr_model.fit(data_train)

        # Train HMM
        lqr_model = HMMLQR(nb_states=states)
        lqr_model.fit(data_train)

        # Pickling the results
        gmr_string = store_path + part + "/gmr/" + 'states_' + str(states) + '.pickle'
        pickle.dump(gmr_model, open(gmr_string, 'wb'))

        lqr_string = store_path + part + "/lqr/" + 'states_' + str(states) + '.pickle'
        pickle.dump(lqr_model, open(lqr_string, 'wb'))

    data_train = data_split_train
    for hmm_u in params_hmm_u:
        print("Training model part %s, param u at %i" % (part, hmm_u))
        # Train HMM
        lqr_model = HMMLQR(gmm_u=hmm_u)
        lqr_model.fit(data_train)

        # Pickling the results
        lqr_string = store_path + part + "/lqr/" + 'u_' + str(hmm_u) + '.pickle'
        pickle.dump(lqr_model, open(lqr_string, 'wb'))
