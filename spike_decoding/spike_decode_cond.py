import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import seaborn as sns
import pandas as pd
import spynal
from spynal.matIO import loadmat
from spynal import info, utils
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

information = ['s1', 's2']
areas = ['PFC']
conditions = ['retain', 'update']
classifier = 'LDA'
n_classes = 4
for condition in conditions:
    for area in areas:
        for type in information:
            data = np.load(f'/Users/huidili/Documents/miller_lab/wmUpdate/150b50s_decode/spk_dec_condition_data/Tir_{type}_{area}_{condition}_feature_label.npz')
            feature = data['feature']
            label = data['label']
            tps = feature.shape[2]
            kf = StratifiedKFold(n_splits=5, shuffle=True)
            decoder = LinearDiscriminantAnalysis(priors=(1/n_classes)*np.ones((n_classes,)))
            accs = np.empty((tps, kf.n_splits))
            for i_fold, (train_idx, test_idx) in enumerate(kf.split(feature, label)):
                for t in range(feature.shape[2]):
                    X_train, X_test = feature[train_idx,:,t], feature[test_idx,:,t]
                    y_train, y_test = label[train_idx], label[test_idx]
                    # compute z-score
                    train_mean = np.mean(X_train)
                    train_sd = np.std(X_train)
                    X_train = (X_train - train_mean) / train_sd
                    X_test = (X_test - train_mean) / train_sd
                    decoder.fit(X_train, y_train)
                    accs[t, i_fold] = decoder.score(X_test, y_test)
            accs_mean = np.mean(accs, axis=1)
            np.savez(f'/Users/huidili/Documents/miller_lab/wmUpdate/150b50s_decode/spk_dec_condition_result/Tir_{type}_{area}_{condition}_{classifier}_accuracy.npz', accuracy=accs_mean)





