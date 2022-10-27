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

def select_trials(trialInfo):
  trial_sel = (trialInfo['correct']) & (~ trialInfo['badTrials']) & (trialInfo['task'] == 'wmUpdate')
  return trial_sel

def select_area(unitInfo):
  area_sel = unitInfo['area']
  adict = {}
  adict['LIP'] = area_sel=='LIP'
  adict['PFC'] = area_sel=='PFC'
  adict['OFC'] = area_sel=='OFC'

  df = pd.DataFrame(adict)

  return df

dir_path = '/Users/huidili/Documents/miller_lab/wmUpdate/mat/'
for session_file in listdir(dir_path):

    trialInfo, spikeTimes, unitInfo = loadmat(dir_path+session_file, variables=['trialInfo', 'spikeTimes', 'unitInfo'], typemap={'trialInfo': 'DataFrame'}, verbose=True)
    session_name = trialInfo['session'][0]

    # group units by area
    area_df = select_area(unitInfo)
    area_df.to_pickle(f'/Users/huidili/Documents/miller_lab/wmUpdate/150b50s_decode/spk_area/{session_name}_spk_area_df.pkl')

    # select trials and compute spiking rate
    trial_idx = select_trials(trialInfo)
    spikeTimes = spikeTimes[trial_idx]
    spike_rate, rate_bins = rate(spikeTimes, method='bin', lims=[-2, 4.5], width=150e-3, step=50e-3)
    np.savez(f'/Users/huidili/Documents/miller_lab/wmUpdate/150b50s_decode/spk_rate/{session_name}_spk_rate.npz', spike_rate=spike_rate, rate_bins=rate_bins)

