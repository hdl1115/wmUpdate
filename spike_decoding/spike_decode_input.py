import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import seaborn as sns
import pandas as pd
import spynal
from spynal.matIO import loadmat
from spynal.spectra import remove_evoked, spectrogram, pool_freq_bands, one_over_f_norm, plot_spectrogram
from os import listdir

def count_number(labels):
    all_label_num = []
    for label in labels:
        each_label_num = []
        for i in range(4):
            each_label_num.append(np.sum(label == i))
        all_label_num.append(np.array(each_label_num).reshape(1,-1))
    return np.concatenate(all_label_num, axis=0)


def create_pseudo_popu(labels_cat, units_cat, min_repeated_trials):
    dec_feature = {}
    dec_label = {}
    info = ['s1', 's2', 'r1', 'r2']
    for type in info:
        dec_feature[type] = []
        dec_label[type] = np.array([0] * min_repeated_trials + [1] * min_repeated_trials + [2] * min_repeated_trials + [
            3] * min_repeated_trials)

    for i, labels in enumerate(labels_cat):
        for m, type in enumerate(info):
            sel_classes_idx = np.zeros_like(labels[m], dtype=bool)  # labels[m]: (trial_num, )
            for n in range(4):
                class_idx = np.where(labels[m] == n)[0]  # trial idx when class number = n
                sel_class_idx = np.random.choice(class_idx, min_repeated_trials, replace=False)
                sel_classes_idx[sel_class_idx] = True
            sel_units = units_cat[i][sel_classes_idx]
            sel_labels = labels[m][sel_classes_idx]
            # sort trials by class
            sort_idx = sel_labels.argsort()
            sort_units = sel_units[sort_idx]
            dec_feature[type].append(sort_units)

    for type in info:
        dec_feature[type] = np.concatenate(dec_feature[type], axis=1)

    return dec_feature, dec_label


# concatenate units and labels
units_cat = []
labels_cat = []
area_all_units = {}
areas = ['PFC', 'LIP', 'OFC']
for area in areas:
    area_all_units[area] = []

n_splits = 5
min_repeated_trials = None

path = '/Users/huidili/Documents/miller_lab/wmUpdate/150b50s_decode/spk_dec_input/'
for filename in listdir(path):
    if ('area' in filename) and ('Tir' in filename):
        # sub_name, session_num, _, _ = filename.split('_')
        # session_name = sub_name+'_'+session_num
        session_name = filename.split('_')[0]
        stationary_unit = np.load(path + session_name + '_stationary.npz')['stationary']
        area_df = pd.read_pickle(path+filename)
        for area in areas:
            area_idx = area_df[area]
            area_all_units[area].append(area_idx[stationary_unit])

        spk_data = np.load(path+session_name+'_spk_rate.npz')['spike_rate']
        label_data = np.load(path+session_name+'_spk_dec_labels.npz')['labels']
        min_labels = np.min(count_number(label_data))
        if min_repeated_trials is None or (min_labels < min_repeated_trials):
            min_repeated_trials = min_labels
        units_cat.append(spk_data[:, stationary_unit])
        labels_cat.append(label_data)

# modify the min so it can be divided by n_splits
min_repeated_trials -= min_repeated_trials % n_splits

for area in areas:
    area_all_units[area] = np.concatenate(area_all_units[area])

dec_feature, dec_label = create_pseudo_popu(labels_cat, units_cat, min_repeated_trials)

info = ['s1', 's2', 'r1', 'r2']
for type in info:
    for area in areas:
        area_idx = area_all_units[area]
        feature = dec_feature[type][:,area_idx]
        label = dec_label[type]
        np.savez(f'/Users/huidili/Documents/miller_lab/wmUpdate/150b50s_decode/spk_dec_data/Tir_{type}_{area}_feature_label.npz', feature=feature, label=label)






