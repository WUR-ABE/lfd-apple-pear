import numpy as np
import os
import pbdlib as pbd
import sklearn as sk
import pickle
import matplotlib.pyplot as plt
import math as m
import time as t
import sys

# Required for rel import
new_path = "/home/agro-legion/thesis_ws/src/includes/pbdlib_custom"

if new_path not in sys.path:
    sys.path.append(new_path)
from gmmr import GMMR
from hmmlqr import HMMLQR

dp = os.path.dirname(pbd.__file__) + '/data/demos/corrected'
filelist = next(os.walk(dp))[2]
data = []
for f in filelist:
    temp = np.genfromtxt(dp + '/' + f, delimiter=',')
    data.append(temp)

select = range(0, len(data[0]),20)
data_short = []
for d in data:
    data_short.append(d[select,:])

data_train, data_test = sk.model_selection.train_test_split(data_short, test_size = 0.2, random_state=1337)

params_gmm = {"nb_states" : [11,10,9,8,7,6,5,4,3,2]}
                
params_hmm = {"nb_states" : [11,10,9,8,7,6,5,4,3,2],
                "gmm_u" : [-5.0, -4.0,-3.0, -2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0]}
params_hmm_state = {"nb_states" : [11,10,9,8,7,6,5,4,3,2]}
params_hmm_u = {"gmm_u" : [-5.0, -4.0,-3.0, -2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0]}
                
#%% GMR
random_gmr = sk.model_selection.GridSearchCV(GMMR(), params_gmm, n_jobs=4, pre_dispatch="n_jobs", cv=4, verbose=3, return_train_score=True)
random_gmr.fit(data_train[0:8])


#%% LQR
lqr_u = sk.model_selection.GridSearchCV(HMMLQR(), params_hmm_u, n_jobs=2, pre_dispatch="n_jobs", cv=4, verbose=3, return_train_score=True)
lqr_u.fit(data_train[0:8])

lqr_state = sk.model_selection.GridSearchCV(HMMLQR(), params_hmm_state, n_jobs=2, pre_dispatch="n_jobs", cv=4, verbose=3, return_train_score=True)
lqr_state.fit(data_train[0:8])

# Determine best model
best_est_gmr = random_gmr.best_estimator_
best_est_lqr_u = lqr_u.best_estimator_
best_est_lqr_state = lqr_state.best_estimator_

gmr_result = random_gmr.cv_results_
lqr_state_result = lqr_state.cv_results_
lqr_u_result = lqr_u.cv_results_

#%% Pickling the results
file_lqr =  open('src/includes/pbdlib_custom/LQR_CV_state.p', 'wb')
file_lqr_2 = open('src/includes/pbdlib_custom/LQR_CV_u.p', 'wb')
file_gmr =  open('src/includes/pbdlib_custom/GMR_CV_short.p', 'wb')

pickle.dump(lqr_state, file_lqr)
pickle.dump(lqr_u, file_lqr_2)
pickle.dump(random_gmr, file_gmr)

#%% Reading the results
gmr1 = pickle.load(open("GMR_CV.p","rb"))
gmr2 = pickle.load(open("src/includes/pbdlib_custom/GMR_CV_2.p", "rb"), encoding="latin1")
lqr = pickle.load(open("src/includes/pbdlib_custom/LQR_CV.p", "rb"), encoding="latin1")

#%% Function for plotting

def plot_test_train(test_mean, train_mean, test_sd, train_sd, labels, title):
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, -1*train_mean, width, label='Train', yerr = train_sd, ecolor = 'lightcoral', capsize=5)
    rects2 = ax.bar(x + width/2, -1*test_mean, width, label='Validation', yerr = test_sd, ecolor = 'lightskyblue', capsize=5)
    #ax.hlines(24, -0.5, 9.5, colors='k')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('states')
    ax.legend()
    
    fig.tight_layout()
    #plt.ylim(0,25)
    plt.show

#%% Quantifying the results of GMR

gmr_result = gmr2.cv_results_
gmr_result = lqr_state_result
gmr_result = lqr_u_result

gmr_test_mean = gmr_result['mean_test_score']
gmr_train_mean = gmr_result['mean_train_score']

gmr_test_sd = gmr_result['std_test_score']
gmr_train_sd = gmr_result['std_train_score']

#plt.bar(params_gmm["nb_states"],-1*test_mean, yerr = test_sd)

gmr_labels = params_gmm["nb_states"]
a = gmr_labels.reverse()
gmr_title = 'Score of cost function for 2 to 11 Gaussian components'

plot_test_train(np.flip(gmr_test_mean), np.flip(gmr_train_mean), np.flip(gmr_test_sd), np.flip(gmr_train_sd), gmr_labels, gmr_title)


#%% Scoring best model
start =t.time()
gmr_score = GMMR.score(gmr2.best_estimator_, data_short)
gmr_time = t.time() - start
#%% Quantifying the results of LQR 
lqr_result = lqr.cv_results_

# means and sd per state
state_lqr_test_mean = []
state_lqr_test_sd = []
state_lqr_train_mean = []
state_lqr_train_sd = []

for i in range(len(params_hmm["nb_states"])):
    testmeansum = 0
    testvarsum = 0
    trainmeansum = 0
    trainvarsum = 0
    
    for j in range(len(params_hmm["gmm_u"])):
        a = i+j*len(params_hmm["nb_states"])
        testmeansum += lqr_result['mean_test_score'][a]
        testvarsum += (lqr_result['std_test_score'][a])**2
        trainmeansum += lqr_result['mean_train_score'][a]
        trainvarsum += (lqr_result['std_train_score'][a])**2
        
    state_lqr_test_mean.append(testmeansum/len(params_hmm["gmm_u"]))
    state_lqr_test_sd.append(m.sqrt(testvarsum/len(params_hmm["gmm_u"])))
    state_lqr_train_mean.append(trainmeansum/len(params_hmm["gmm_u"]))
    state_lqr_train_sd.append(m.sqrt(trainvarsum/len(params_hmm["gmm_u"])))

state_lqr_test_mean = np.array(state_lqr_test_mean)
state_lqr_test_sd = np.array(state_lqr_test_sd)
state_lqr_train_mean = np.array(state_lqr_train_mean)
state_lqr_train_sd = np.array(state_lqr_train_sd)


# means and sd per gmm_u
u_lqr_test_mean = []
u_lqr_test_sd = []
u_lqr_train_mean = []
u_lqr_train_sd = []

for i in range(len(params_hmm["gmm_u"])):
    testmeansum = 0
    testvarsum = 0
    trainmeansum = 0
    trainvarsum = 0
    
    for j in range(len(params_hmm["nb_states"])):
        a = i*len(params_hmm["nb_states"])+j
        testmeansum += lqr_result['mean_test_score'][a]
        testvarsum += (lqr_result['std_test_score'][a])**2
        trainmeansum += lqr_result['mean_train_score'][a]
        trainvarsum += (lqr_result['std_train_score'][a])**2
        
    u_lqr_test_mean.append(testmeansum/len(params_hmm["nb_states"]))
    u_lqr_test_sd.append(m.sqrt(testvarsum/len(params_hmm["nb_states"])))
    u_lqr_train_mean.append(trainmeansum/len(params_hmm["nb_states"]))
    u_lqr_train_sd.append(m.sqrt(trainvarsum/len(params_hmm["nb_states"]))) 
    
u_lqr_test_mean = np.array(u_lqr_test_mean)
u_lqr_test_sd = np.array(u_lqr_test_sd)
u_lqr_train_mean = np.array(u_lqr_train_mean)
u_lqr_train_sd = np.array(u_lqr_train_sd)


#%% plot LQR

state_lqr_labels = params_hmm["nb_states"]
#a = state_lqr_labels.reverse()
state_lqr_title = 'Score of cost function for 2 to 11 states'
plot_test_train(state_lqr_test_mean, state_lqr_train_mean, state_lqr_test_sd, state_lqr_train_sd, state_lqr_labels, state_lqr_title)
#%% 
u_lqr_labels = params_hmm["gmm_u"]
a = u_lqr_labels.reverse()
u_lqr_title = 'Score of cost function for rho ranging form -5 to 5'
plot_test_train(u_lqr_test_mean, u_lqr_train_mean, u_lqr_test_sd, u_lqr_train_sd, u_lqr_labels, u_lqr_title)

#%% Comprative plot between LQR rho's and components of GMR

fig, ax = plt.subplots(nrows=1, ncols=2)

x1 = np.arange(len(u_lqr_labels))
x2 = np.arange(len(gmr_labels))
width = 0.6

rects1 = ax[0].bar(x1, -1*u_lqr_test_mean, width, label='HMM with LQT', yerr = u_lqr_test_sd, color='C1', ecolor = 'lightskyblue', capsize=5)
ax[0].hlines(24, -0.5, 10.5, colors='k')
ax[0].set_ylabel('Scores')
ax[0].set_xticks(x1)
ax[0].set_xticklabels(u_lqr_labels)
ax[0].legend(labels = ("HMM with LQT","GMM with GMR"))
ax[0].set_xlabel('rho')
rects2 = ax[1].bar(x2, -1*gmr_test_mean, width, label='GMM with GMR', yerr = gmr_test_sd, color='C0', ecolor = 'lightcoral', capsize=5)
ax[1].hlines(24, -0.5, 9.5, colors='k')
ax[1].set_xticks(x2)
ax[1].set_xticklabels(gmr_labels)
ax[1].set_xlabel('Gaussian component')
ax[0].legend(handles = (rects1, rects2), labels = ("HMM with LQT","GMM with GMR"), loc = 'center left')
fig.suptitle("Comparison between scores of GMM with GMR and the HMM with LQT")
#ax[1].legend()
plt.ylim(0,25)
plt.show


#%% Scoring best LQR
start =t.time()
lqr_score = HMMLQR.score(lqr.best_estimator_, data_short)
lqr_time = t.time() - start
#%%

# means and sd per state
state_time_mean = []
state_time_sd = []

for i in range(len(params_hmm["nb_states"])):
    testmeansum = 0
    testvarsum = 0
    
    for j in range(len(params_hmm["gmm_u"])):
        a = i+j*len(params_hmm["nb_states"])
        testmeansum += lqr_result['mean_fit_time'][a]/60
        testvarsum += (lqr_result['std_fit_time'][a]/60)**2

        
    state_time_mean.append(testmeansum/len(params_hmm["gmm_u"]))
    state_time_sd.append(m.sqrt(testvarsum/len(params_hmm["gmm_u"])))

state_time_mean = np.array(state_time_mean)
state_time_sd = np.array(state_time_sd)

# means and sd per u
u_time_mean = []
u_time_sd = []

for i in range(len(params_hmm["gmm_u"])):
    testmeansum = 0
    testvarsum = 0
    
    for j in range(len(params_hmm["nb_states"])):
        a = i*len(params_hmm["nb_states"])+j
        testmeansum += lqr_result['mean_fit_time'][a]/60
        testvarsum += (lqr_result['std_fit_time'][a]/60)**2

        
    u_time_mean.append(testmeansum/len(params_hmm["gmm_u"]))
    u_time_sd.append(m.sqrt(testvarsum/len(params_hmm["gmm_u"])))

u_time_mean = np.array(u_time_mean)
u_time_sd = np.array(u_time_sd)

#%% Some ANOVA
time_std = lqr_result['std_score_time']
time = lqr_result['mean_score_time']
time_var = time_std**2
time_mean = np.mean(time)

Between_SS = sum(44*(time-time_mean)**2)
Within_SS = sum(43*time_var)

Total_SS = Between_SS+Within_SS
Between_df = 9
Within_df = 430
Total_df = 339
Between_MS = Between_SS/Between_df
Within_MS = Within_SS/Within_df
Total_MS = Total_SS/Total_df
F = Between_MS/Within_MS

## Make traj with only start and end point
start_end = np.vstack([data_test[0][0,:],data_test[0][-1,:]])

#%% Predict trajs and plot

gmr_pred = GMMR.predict(gmr2.best_estimator_, [start_end])[0]
lqr_pred = HMMLQR.predict(lqr.best_estimator_, [start_end], hor=204)[0]

#%% plotting predictions
fig, ax = plt.subplots(nrows=6)
fig.set_size_inches(12,15)
for i in range(6):
    for p in lqr_pred:
        ax[i].plot(p[:, i])

#%%

# plotting
fig, ax = plt.subplots(ncols=3)
Y = 2
X = 1
for p in gmr_pred[0:5]:
    ax[0].plot(p[:, X], p[:, Y])
    
for p in data_test[0:5]:
    ax[1].plot(p[:, X], p[:, Y])
    
for p in lqr_pred[0:5]:
    ax[2].plot(p[:, X], p[:, Y])

    
ax[1].set_xlabel("Y (m)", fontsize=12)
ax[0].set_ylabel("Z (m)", fontsize=12)
ax[0].set_title("GMM with GMR")
ax[1].set_title("Demonstration")
ax[2].set_title("HMM with LQT")
fig.suptitle("Y-Z plot of 5 demonstrated trajectories", fontsize=14)
