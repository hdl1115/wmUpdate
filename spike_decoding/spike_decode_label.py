import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import seaborn as sns
import pandas as pd
import spynal
from spynal.matIO import loadmat
from spynal.spectra import remove_evoked, spectrogram, pool_freq_bands, one_over_f_norm, plot_spectrogram
from os import listdir

def select_trials(trialInfo):
  trial_sel = (trialInfo['correct']) & (~ trialInfo['badTrials']) & (trialInfo['task'] == 'wmUpdate')
  return trial_sel

path = '/Users/huidili/Documents/miller_lab/wmUpdate/mat/'
for filename in listdir(path):
    trialInfo = loadmat(path+filename, variables=['trialInfo'], typemap={'trialInfo': 'DataFrame'}, verbose=True)
    session_name = trialInfo['session'][0]
    trial_idx = select_trials(trialInfo)
    s_dict = {45:0, 135:1, 225:2, 315:3}
    r_dict = {1:0, 5:1, 6:1, 8:2, 9:2, 10:3, 13:3}
    s1_labels = np.array([s_dict[loc] for loc in trialInfo['location1'][trial_idx]]).reshape(1,-1)
    s2_labels = np.array([s_dict[loc] for loc in trialInfo['location2'][trial_idx]]).reshape(1,-1)
    r1_labels = np.array([r_dict[rwd] for rwd in trialInfo['offer1'][trial_idx]]).reshape(1,-1)
    r2_labels = np.array([r_dict[rwd] for rwd in trialInfo['offer2'][trial_idx]]).reshape(1,-1)
    labels_cat = np.concatenate((s1_labels, s2_labels, r1_labels, r2_labels), axis=0)
    np.savez(f'/Users/huidili/Documents/miller_lab/wmUpdate/spk_dec_label/{session_name}_spk_dec_labels.npz', labels=labels_cat)


