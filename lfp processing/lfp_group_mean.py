import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import seaborn as sns
import pandas as pd
from scipy import stats
import spynal
from spynal.matIO import loadmat
from spynal.spectra import remove_evoked, spectrogram, pool_freq_bands, one_over_f_norm, plot_spectrogram
from os import listdir

def lfp_plot(power_z, t_wave, f_wave, condition, area_name):
    fig = plt.figure(figsize=(15, 5))
    fig, ax1 = plt.subplots(figsize=(10,4))
    # sns.heatmap(power_z, cmap='jet', vmin=-15, vmax=15, ax=ax1, cbar_kws={'label':'t-statistics'})
    sns.heatmap(power_z, cmap='jet', vmin=-1, vmax=4, ax=ax1, cbar_kws={'label':'Z-score'})
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
    ax1.vlines(x=ax1_vlines, ymin=ax1_b, ymax=ax1_t, linestyles='--', colors='black', lw=1)  # S1 start line
    for i in range(len(vline_text)):
        text(ax1_vlines[i * 2], ax1_t + 1, vline_text[i], verticalalignment='center')
    ax1.set_title(f'{condition} LFP power spectrogram in {area_name} for Tiergan', pad=25)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')

    plt.legend()
    plt.tight_layout()
    fig.savefig(f'/Users/huidili/Documents/miller_lab/wmUpdate/lfp_plot/{condition}_mean_Tir_{area_name}.png')

    plt.show()

def lfp_tstats_plot(tstats, t_wave, f_wave, area_name):
    fig, ax1 = plt.subplots(figsize=(10,4))
    sns.heatmap(tstats, cmap='jet', vmin=-15, vmax=15, ax=ax1, cbar_kws={'label':'t-statistics'})
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
    ax1.vlines(x=ax1_vlines, ymin=ax1_b, ymax=ax1_t, linestyles='--', colors='black', lw=1)  # S1 start line
    for i in range(len(vline_text)):
        text(ax1_vlines[i * 2], ax1_t + 1, vline_text[i], verticalalignment='center')
    ax1.set_title(f'Update - Retain LFP power t-statistics in {area_name} for Tiergan', pad=25)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')

    plt.tight_layout()
    fig.savefig(f'/Users/huidili/Documents/miller_lab/wmUpdate/lfp_plot/tstats_Tir_{area_name}.png')

    plt.show()


path = '/Users/huidili/Documents/miller_lab/wmUpdate/lfp_output/'
f_wave = np.empty(0)
t_wave = np.empty(0)
retain_power_elec = {}
update_power_elec = {}
retain_band_elec = {}
update_band_elec = {}
areas = ['PFC', 'LIP', 'OFC']
bands = np.array([[3,8], [10,32], [40,100]])
band_names = ['theta', 'beta', 'gamma']
for area in areas:
    retain_power_elec[area] = []
    update_power_elec[area] = []
    retain_band_elec[area] = []
    update_band_elec[area] = []

for filename in listdir(path):
    if filename.endswith('.pkl') and ('Tir' in filename):
        area_df = pd.read_pickle(path+filename)
        # sub_name, session_num, _, _ = filename.split('_')
        # session_name = sub_name+'_'+session_num
        session_name = filename.split('_')[0]
        retain_data = np.load(path+session_name+'_retain_lfpproc.npz')
        update_data = np.load(path+session_name+'_update_lfpproc.npz')
        t_wave = retain_data['t_wave']
        f_wave = retain_data['f_wave']
        # accumulate power across all electrodes
        for area in areas:
            elec_idx = area_df[area]
            retain_power = np.mean(retain_data['power'], axis=-1)[:,:,elec_idx]
            update_power = np.mean(update_data['power'], axis=-1)[:,:,elec_idx]
            retain_power_elec[area].append(retain_power)
            update_power_elec[area].append(update_power)
            # retain_band = pool_freq_bands(retain_power, bands, axis=0, freqs=f_wave)
            # update_band = pool_freq_bands(update_power, bands, axis=0, freqs=f_wave)
            # retain_band_elec[area].append(retain_band)
            # update_band_elec[area].append(update_band)


for area in areas:
    retain_all_elec = np.concatenate(retain_power_elec[area], axis=-1)
    update_all_elec = np.concatenate(update_power_elec[area], axis=-1)
    retain_elec_mean = np.mean(retain_all_elec, axis=-1)
    update_elec_mean = np.mean(update_all_elec, axis=-1)
    t, p = stats.ttest_rel(update_all_elec, retain_all_elec, axis=-1)
    # retain_band_all_elec = np.concatenate(retain_band_elec[area], axis=-1)
    # update_band_all_elec = np.concatenate(update_band_elec[area], axis=-1)
    # np.savez(f'/Users/huidili/Documents/miller_lab/wmUpdate/lfp_plot/condition_band_{area}_ISA.npz', rb=retain_band_all_elec, ub=update_band_all_elec)



    lfp_plot(retain_elec_mean, t_wave, f_wave, 'retain', area)
    lfp_plot(update_elec_mean, t_wave, f_wave, 'update', area)
    lfp_tstats_plot(t, t_wave, f_wave, area)










# # data: power, t_wave, f_wave, power_band
# session_name = '061311Tir'
# retain_data = np.load(f'{session_name}_retain_lfpproc.npz')
# update_data = np.load(f'{session_name}_update_lfpproc.npz')

# average all electrodes across trials

