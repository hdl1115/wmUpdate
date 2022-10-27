import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import seaborn as sns
import pandas as pd
from scipy import stats
import spynal
from spynal.matIO import loadmat
from spynal.spikes import times_to_bool, rate
from spynal.spectra import remove_evoked, spectrogram, pool_freq_bands, one_over_f_norm, plot_spectrogram
from os import listdir

def remove_temporal_trends(spikeTimes):
    trial_num = spikeTimes.shape[0]
    unit_num = spikeTimes.shape[1]
    bin_size = 20
    step = 1
    total_bins = int((trial_num - bin_size) / step + 1)
    tps = 6.5
    stationary_check = np.ones(unit_num, dtype=bool)
    for i in range(unit_num):
        spkTimes_unit = spikeTimes[:,i] # (trial_num, )
        compute_rate = np.vectorize(lambda x:np.size(x)/tps)
        spk_rate_unit = compute_rate(spkTimes_unit)
        var_unit = np.var(spk_rate_unit)
        all_var_bin = []
        for j in range(0, total_bins, step):
            var_bin = np.var(spk_rate_unit[j:j+bin_size])
            all_var_bin.append(var_bin)
        print(all_var_bin)
        print(var_unit)
        if np.mean(all_var_bin) < var_unit/2:
            stationary_check[i] = 0

    return stationary_check

dir_path = '/Users/huidili/Documents/miller_lab/wmUpdate/mat/'
for session_file in listdir(dir_path):
    trialInfo, spikeTimes, unitInfo = loadmat(dir_path + session_file,
                                              variables=['trialInfo', 'spikeTimes', 'unitInfo'],
                                              typemap={'trialInfo': 'DataFrame'}, verbose=True)
    session_name = trialInfo['session'][0]
    np.savez(f'/Users/huidili/Documents/miller_lab/wmUpdate/unit_stationary/{session_name}_stationary.npz', stationary=remove_temporal_trends(spikeTimes))


