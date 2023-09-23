import csv
import random
import math
import operator
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from matplotlib.ticker import StrMethodFormatter

def load_dataset(filename):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
    return dataset
    
def getUniqueWords(dataset):
    unique_words = list()
    for x in range(len(dataset)):
        for y in range(len(dataset[1])):
            if dataset[x][y] not in unique_words:
                unique_words.append(dataset[x][y])
    return unique_words
    
def get_accuracy(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual))

def get_confusion_matrix(actual, pred):
    K = len(np.unique(actual))
    output = np.zeros((K, K)).astype(int)
    l = preprocessing.LabelEncoder()
    l.fit(np.unique(actual))      
    for i in range(len(actual)):
        x = l.transform([actual[i]])
        y = l.transform([pred[i]])
        output[x[0]][y[0]] += 1
    return output


dataset = load_dataset("./datasets/car.csv") 
unique_words = getUniqueWords(dataset);

le = preprocessing.LabelEncoder()
le.fit(unique_words)
transformedDataset = [le.transform(dataset[i]) for i in range(len(dataset))] 
datas = [x[0:len(transformedDataset[1])-1] for x in transformedDataset]
labels = [x[-1] for x in transformedDataset]
datas, labels = shuffle(datas, labels)
X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.2)

confusion_matrix_vector = list()
accuracy_vector = list()
clf_vector = list()
for i in range(5):
    max_depth = i+1
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confusion_matrix_vector.append(get_confusion_matrix(y_test, y_pred))
    accuracy_vector.append(get_accuracy(y_test, y_pred))
    clf_vector.append(clf)
#print(accuracy_vector)
plt.figure()
plt.plot([x+1 for x in range(len(accuracy_vector))] ,accuracy_vector,'-')
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0%}'))
plt.xlabel("Depth of Tree")
plt.ylabel("Accuracy (%)")  
plt.ylim((0,1))
plt.title('Decision Tree Classifier')
plt.show()

maxpos = accuracy_vector.index(max(accuracy_vector))
print("The best depth between these is ", maxpos + 1, "\n") 

print("At this depth, confusion matrix is: ")
print(confusion_matrix_vector[maxpos], "\n")

print("At this depth, accuracy is: ")
print(accuracy_vector[maxpos])

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf_vector[maxpos])
fig.savefig('decisionTree.png')