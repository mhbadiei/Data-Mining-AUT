import csv
import random
import math
import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score


def loadDataset(filename):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(len(dataset[1])):
                dataset[x][y] = float(dataset[x][y])
    return dataset

def make_meshgrid(x, y, h=.01):
    x_min, x_max = min(x) - 0.3, max(x) + 0.3
    y_min, y_max = min(y) - 0.3, max(y) + 0.3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

dataset = loadDataset("./datasets/binary_2d.csv") 
#print(len(dataset),len(dataset[1]))

X_train, X_test, y_train, y_test = train_test_split([x[0:len(dataset[1])-1] for x in dataset], [x[-1] for x in dataset], test_size=0.1)
#print(len(X_train),len(X_train[1]),len(X_test),len(X_test[1]))

SVM = SVC(kernel = 'linear')
SVMClassifier = SVM.fit(X_train, y_train)
predictions = SVMClassifier.predict(X_test)
a = accuracy_score(y_test, predictions)
p = precision_score(y_test, predictions, average = 'weighted')
r = recall_score(y_test, predictions, average = 'weighted')
print(a, p, r)

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of linear SVC ')
# Set-up grid for plotting.
X0, X1 = [x[0] for x in X_train], [x[1] for x in X_train]
print(X0)
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, SVMClassifier, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_ylabel('y label here')
ax.set_xlabel('x label here')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()
