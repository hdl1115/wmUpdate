import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import seaborn as sns
import pandas as pd
import spynal
from spynal.matIO import loadmat
from spynal.spectra import remove_evoked, spectrogram, pool_freq_bands, one_over_f_norm, plot_spectrogram

## data selection
def select_trials(trialInfo):
  trial_sel = (trialInfo['correct']) & (~ trialInfo['badTrials']) & (trialInfo['task'] == 'wmUpdate')
  return trial_sel

def select_electrode(electroInfo):
  has_lfp = electroInfo['hasLFP']
  num_units = electroInfo['numUnits']
  units_lfp = num_units[has_lfp] # number of units for electrodes with LFP
  elec_sel = units_lfp > 0
  return elec_sel

def select_area(electroInfo):
  units_flp_idx = electroInfo['hasLFP'] & (electroInfo['numUnits'] > 0)
  area_sel = electroInfo['area'][units_flp_idx]
  adict = {}
  adict['LIP'] = area_sel=='LIP'
  adict['PFC'] = area_sel=='PFC'
  adict['OFC'] = area_sel=='OFC'

  df = pd.DataFrame(adict)

  return df


def select_groups(trialInfo):
    trial_idx = select_trials(trialInfo)
    # for Tiergan: r1=10, r2=8, r3=5, r4=1
    # for Isaura: r1=13, r2=9, r3=6, r4=1
    if 'Tir' in trialInfo['session'][0]:
        r2 = 8
        r3 = 5
    else:
        r2 = 9
        r3 = 6

    offer1 = trialInfo['offer1'][trial_idx]
    offer2 = trialInfo['offer2'][trial_idx]
    larger_ordinal = trialInfo['largerOrdinal'][trial_idx]

    gdict = {}
    gdict['reward2_retain'] = (offer1 == r2) & (larger_ordinal == 1)
    gdict['reward2_update'] = (offer1 == r2) & (larger_ordinal == 2)
    gdict['reward3_retain'] = (offer1 == r3) & (larger_ordinal == 1)
    gdict['reward3_update'] = (offer1 == r3) & (larger_ordinal == 2)

    df = pd.DataFrame(gdict)

    return df

## lfp processing
def lfp_processing(lfp):
    # wavelet power spectrogram
    power, f_wave, t_wave = spectrogram(lfp, smp_rate=1000, axis=0, method='wavelet', spec_type='power',
                                        freqs=2 ** np.arange(1, 7.2, 0.2), downsmp=10)
    # convert to actual times
    lfp_tps = lfpSchema['index'][0]
    t_wave += lfp_tps[0]
    # compute base10 log of power
    power = np.log10(power)
    # compute z-score
    power = spynal.utils.zscore(power, axis=3, time_range=np.array([-0.6, 0]), time_axis=1, timepts=t_wave)
    # pool wavelet spectrograms by frequency band (theta, beta, and gamma)
    bands = np.array([[20,35], [50,120]])
    power_band = pool_freq_bands(power, bands, axis=0, freqs=f_wave)
    return power, t_wave, f_wave, power_band

def lfp_plot(power_z, t_wave, f_wave, power_band, group_name, area_idx):
    # ax1 spectrogram
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    mean_power = np.mean(power_z[:, :, area_idx], axis=(2, 3))
    sns.heatmap(mean_power, cmap='jet', ax=ax1, cbar_kws={'label':'LFP power (z-score)'})
    ax1.set_xticks(np.arange(0, 650, 100))
    ax1.set_xticklabels(np.arange(-2, 5, 1), rotation=0)
    ax1.invert_yaxis()
    ylabels = 2 ** np.arange(1, 8)
    ax1.set_yticks(np.linspace(0, 30, len(ylabels)))
    ax1.set_yticklabels(ylabels, rotation=0)

    # add vlines
    ax1_b, ax1_t = ax1.get_ylim()
    ax1_vlines = [200, 220, 280, 300, 360, 380, 440, 460]
    vline_text = ['S1', 'R1', 'S2', 'R2']
    ax1.vlines(x=ax1_vlines, ymin=ax1_b, ymax=ax1_t, linestyles='--', colors='w', lw=1)  # S1 start line
    for i in range(len(vline_text)):
        text(ax1_vlines[i * 2], ax1_t + 1, vline_text[i], verticalalignment='center')
    ax1.set_title('wavelet spectrogram', pad=25)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')

    # ax2 band plot
    ax2 = fig.add_subplot(1, 2, 2)
    band_names = ['beta', 'gamma']
    for i_band in range(2):
        mean = np.mean(power_band[i_band], axis=(1, 2))
        ax2.plot(t_wave, mean, '-', color='C%d' % i_band, label=band_names[i_band])
        # std = np.std(power_band[i_band], axis=(1, 2))
        # ax2.fill_between(t_wave, mean - std, mean + std, alpha=0.2)
    ax2_b, ax2_t = ax2.get_ylim()
    ax2.set_ylim(ax2_b, ax2_t)
    ax2.set_xlim(-2, 4.5)
    ax2_vlines = [0, 0.2, 0.8, 1, 1.6, 1.8, 2.4, 2.6]
    for i in range(0, 8, 2):
        ax2.axvspan(ax2_vlines[i], ax2_vlines[i + 1], alpha=0.2, color='gray', linewidth=0)
    for i in range(len(vline_text)):
        text(ax2_vlines[i * 2], ax2_t + 0.2, vline_text[i], verticalalignment='center')
    ax2.set_title('band power', pad=25)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('LFP Power (z-score)')

    plt.legend()
    fig.savefig(f'/Users/huidili/Documents/miller lab/lfp_plot_{group_name}.png')
    # plt.show()

if __name__ == '__main__':
    path = '/Users/huidili/Documents/miller lab/061411Tir.mat'
    trialInfo, electroInfo, lfp, lfpSchema, = loadmat(path, variables=['trialInfo', 'electrodeInfo',
                                                                                        'lfp',
                                                                                        'lfpSchema'],
                                                                             typemap={'trialInfo': 'DataFrame'},
                                                                             verbose=True)
    # select trials and electrodes and apply selection to all data
    elec_idx = select_electrode(electroInfo)
    trial_idx = select_trials(trialInfo)
    lfp = lfp[:,:,trial_idx][:,elec_idx]

    # remove evoked potential
    lfp = remove_evoked(lfp, axis=-1, method='mean', design=trialInfo['condition'][trial_idx])

    group_df = select_groups(trialInfo)
    area_df = select_area(electroInfo)

    # lfp processing
    group_name = ['reward2_retain', 'reward2_update', 'reward3_retain', 'reward3_update']

    for i in range(4):
        power, t_wave, f_wave, power_band = lfp_processing(lfp[:,:,group_df[group_name[i]]])
        lfp_plot(power, t_wave, f_wave, power_band, group_name[i], area_df['PFC'])