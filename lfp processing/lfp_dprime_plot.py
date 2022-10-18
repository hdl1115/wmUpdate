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

def lfp_plot(dprime_mean, t_wave, f_wave, area_name):
    fig, ax1 = plt.subplots(figsize=(10,4))
    sns.heatmap(dprime_mean, cmap='jet', ax=ax1, vmin=-0.15, vmax=0.15, cbar_kws={'label':'d-prime'})
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
    ax1.set_title(f'Update - Retain mean d-prime in {area_name} for Isaura', pad=25)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')


    plt.tight_layout()
    fig.savefig(f'/Users/huidili/Documents/miller_lab/wmUpdate/lfp_plot/dprime_mean_ISA_{area_name}.png')

    plt.show()

def dprime_tstats_plot(tstats, t_wave, f_wave, area_name):
    fig, ax1 = plt.subplots(figsize=(10,4))
    sns.heatmap(tstats, cmap='jet', ax=ax1, vmin=-15, vmax=15, cbar_kws={'label':'t-statistic'})
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
    ax1.set_title(f'Update - Retain d-prime t-statistic in {area_name} for Isaura', pad=25)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')


    plt.tight_layout()
    fig.savefig(f'/Users/huidili/Documents/miller_lab/wmUpdate/lfp_plot/dprime_tstats_ISA_{area_name}.png')

    plt.show()

data_ft = np.load('/Users/huidili/Documents/miller_lab/wmUpdate/lfp_output/061311Tir_retain_lfpproc.npz')
f_wave = data_ft['f_wave']
t_wave = data_ft['t_wave']
areas = ['PFC', 'LIP', 'OFC']
for area in areas:
    dprime = np.load(f'/Users/huidili/Documents/miller_lab/wmUpdate/lfp_dprime/ISA_{area}_lfp_dprime.npy')
    dprime_elec_mean = np.mean(dprime, axis=-1)
    t, p = stats.ttest_1samp(dprime, 0, axis=-1)

    dprime_tstats_plot(t, t_wave, f_wave, area)



