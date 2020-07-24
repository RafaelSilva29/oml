#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:56:46 2020

@author: GrupoA
"""

import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Perceptron
from sklearn.ensemble import AdaBoostClassifier

warnings.filterwarnings("ignore")


def plot_cross_val_score(data, labels):
    y_pos = np.arange(len(labels))    
    plt.bar(y_pos, data, align='center', alpha=0.5)
    plt.xticks(y_pos, labels)
    plt.ylabel('Accuracy')
    plt.xlabel('Numbers')
    plt.title('Cross Validation Score - OneVsAll')
    plt.show()


###################################################################################################
################################# Load Data #######################################################

digits = load_digits()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target,test_size=0.2, random_state=0)

# Train
threshold_train = np.where((y_train == 0) | (y_train == 1) | (y_train == 7) | (y_train == 8))
y_train_thres, x_train_thres = y_train[threshold_train], x_train[threshold_train]

# Test
threshold_test = np.where((y_test == 0) | (y_test == 1) | (y_test == 7) | (y_test == 8))
y_test_thres, x_test_thres = y_test[threshold_test], x_test[threshold_test]


###################################################################################################
################################# Training a classifier (4  numbers) ##############################

num_iter = 5

start_time_OVA = time.time()


OVA = OneVsRestClassifier(Perceptron(max_iter=num_iter, random_state=0))
OVA.fit(x_train_thres, y_train_thres)
predictionsOVA = OVA.predict(x_test_thres)
scoreOVA = OVA.score(x_test_thres, y_test_thres)


cmOVA = metrics.confusion_matrix(y_test_thres, predictionsOVA)
plt.figure(figsize=(9,9))
sns.heatmap(cmOVA, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'OneVsAll - Accuracy Score: {0}'.format(scoreOVA)
plt.title(all_sample_title, size = 15);
plt.show()


finish_time_OVA = time.time() - start_time_OVA


###################################################################################################
################################# Training a classifier (all  numbers) ############################

perceptron = cross_val_score(OneVsRestClassifier(Perceptron(max_iter=num_iter)), 
                             digits.data, digits.target,
                             scoring='accuracy', cv=10)

plot_cross_val_score(perceptron, digits.target_names)


###################################################################################################
################################# Results  ########################################################

print('#'*10 + ' Results OneVsAll ' + '#'*10)
print("\nScore OneVsAll: %.5f%%" % (scoreOVA*100))
print("Time OneVsAll: %.5fs" % finish_time_OVA)
