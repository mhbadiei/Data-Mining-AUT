import csv
import random
import math
import operator
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

def loadDataset(filename):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        #for x in range(len(dataset)-1):
        #    for y in range(19):
        #        dataset[x+1][y+1] = float(dataset[x+1][y+1])
    return dataset
    
def getUniqueWords(dataset):
    unique_words = list()
    for x in range(len(dataset)):
        for y in range(len(dataset[1])):
            if dataset[x][y] not in unique_words:
                unique_words.append(dataset[x][y])
    return unique_words
    

dataset = loadDataset("./datasets/car.csv") 
unique_words = getUniqueWords(dataset);

le = preprocessing.LabelEncoder()
le.fit(unique_words)
transformedDataset = [le.transform(dataset[i]) for i in range(len(dataset))] 
#print(transformedDataset)
#print(unique_words)
#print(len(dataset[1]))
#print(le.transform(dataset[1]) )
#print([x[0:len(dataset[1])-1] for x in dataset])
#print([x[-1] for x in dataset])

X_train, X_test, y_train, y_test = train_test_split([x[0:len(transformedDataset[1])-1] for x in transformedDataset], [x[-1] for x in transformedDataset], test_size=0.2)

#print(len(X_train))
#print(len(X_train[1]))

max_depth = 2
clf = tree.DecisionTreeClassifier(max_depth=max_depth)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf)
fig.savefig('imagename.png')