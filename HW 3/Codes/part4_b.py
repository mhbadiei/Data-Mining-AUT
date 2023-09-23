import csv
import random
import math
import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def loadDataset(filename):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(len(dataset[1])):
                dataset[x][y] = float(dataset[x][y])
    return dataset

def get_accuracy(actual, predicted):
    truePrediction = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            truePrediction += 1
    return 100 * truePrediction / float(len(actual))

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


model = SVC(kernel = 'linear').fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = get_accuracy(y_test, pred)
print("Accuracy of SVM classifier for the test datas is ", accuracy)
plot_list = [accuracy]

plt.figure(figsize=(8,8))
colors_list = ['Red']
graph = plt.bar(['Accuracy of Test'], plot_list, color = colors_list)
plt.title('Accuracy of SVM Classifier')

i = 0
for p in graph:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    plt.text(x+width/2,
             y+height*1.01,
             str(plot_list[i])+'%',
             ha='center',
             weight='bold')
    i+=1

fig, ax = plt.subplots()
title = ('SVM (Linear SVC) Classifier ')
X0, X1 = [x[0] for x in X_train], [x[1] for x in X_train]
xx, yy = make_meshgrid(X0, X1)
plot_contours(ax, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
plt.show()
