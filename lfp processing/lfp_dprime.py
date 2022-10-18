import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import seaborn as sns
import pandas as pd
from scipy import stats
import spynal
from spynal import info
from spynal.matIO import loadmat
from spynal.spectra import remove_evoked, spectrogram, pool_freq_bands, one_over_f_norm, plot_spectrogram
from os import listdir

path = '/Users/huidili/Documents/miller_lab/wmUpdate/lfp_output/'

f_wave = np.empty(0)
t_wave = np.empty(0)
elec_dprime = {}
areas = ['PFC', 'LIP', 'OFC']
for area in areas:
    elec_dprime[area] = []

for filename in listdir(path):
    if filename.endswith('.pkl') and ('ISA' in filename):
        area_df = pd.read_pickle(path+filename)
        sub_name, session_num, _, _ = filename.split('_')
        session_name = sub_name+'_'+session_num
        # session_name = filename.split('_')[0]
        retain_data = np.load(path+session_name+'_retain_lfpproc.npz')
        update_data = np.load(path+session_name+'_update_lfpproc.npz')
        t_wave = retain_data['t_wave']
        f_wave = retain_data['f_wave']
        # accumulate power across all electrodes
        for area in areas:
            elec_idx = area_df[area]
            retain_power = retain_data['power'][:,:,elec_idx]
            update_power = update_data['power'][:,:,elec_idx]
            elec_dprime[area].append(info.neural_info_2groups(update_power, retain_power, axis=-1, method='dprime', keepdims=False))

subject_name = 'ISA'
for area in areas:
    all_elec_dprime = np.concatenate(elec_dprime[area], axis=-1)
    print(all_elec_dprime.shape)
    np.save(f'/Users/huidili/Documents/miller_lab/wmUpdate/lfp_dprime/{subject_name}_{area}_lfp_dprime.npy', all_elec_dprime)
