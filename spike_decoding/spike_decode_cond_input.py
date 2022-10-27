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
    info = ['s1', 's2']
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


# concatenate units and labels for two conditions
retain_units_cat = []
update_units_cat = []
retain_labels_cat = []
update_labels_cat = []
retain_label_num = []
update_label_num = []
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
            area_all_units[area].append(area_df[area][stationary_unit])
        condition_df = pd.read_pickle(path+session_name+'_condition_df.pkl')
        spk_data = np.load(path+session_name+'_spk_rate.npz')
        label_data = np.load(path+session_name+'_spk_dec_labels.npz')
        retain_trial_idx = condition_df['retain']
        update_trial_idx = condition_df['update']
        retain_units_cat.append(spk_data['spike_rate'][retain_trial_idx][:, stationary_unit])
        update_units_cat.append(spk_data['spike_rate'][update_trial_idx][:, stationary_unit])
        retain_labels = label_data['labels'][:, retain_trial_idx]
        update_labels = label_data['labels'][:, update_trial_idx]
        min_labels = min(np.min(count_number(retain_labels[:2])), np.min(count_number(update_labels[:2])))
        if min_repeated_trials is None or (min_labels < min_repeated_trials):
            min_repeated_trials = min_labels
        retain_labels_cat.append(retain_labels)
        update_labels_cat.append(update_labels)

# modify the min so it can be divided by n_splits
min_repeated_trials -= min_repeated_trials % n_splits

for area in areas:
    area_all_units[area] = np.concatenate(area_all_units[area])

retain_dec_feature, retain_dec_label = create_pseudo_popu(retain_labels_cat, retain_units_cat, min_repeated_trials)
update_dec_feature, update_dec_label = create_pseudo_popu(update_labels_cat, update_units_cat, min_repeated_trials)

info = ['s1', 's2']
for type in info:
    for area in areas:
        area_idx = area_all_units[area]
        retain_feature = retain_dec_feature[type][:,area_idx]
        retain_label = retain_dec_label[type]
        update_feature = update_dec_feature[type][:, area_idx]
        update_label = update_dec_label[type]
        np.savez(
            f'/Users/huidili/Documents/miller_lab/wmUpdate/150b50s_decode/spk_dec_condition_data/Tir_{type}_{area}_retain_feature_label.npz',
            feature=retain_feature, label=retain_label)
        np.savez(
            f'/Users/huidili/Documents/miller_lab/wmUpdate/150b50s_decode/spk_dec_condition_data/Tir_{type}_{area}_update_feature_label.npz',
            feature=update_feature, label=update_label)







