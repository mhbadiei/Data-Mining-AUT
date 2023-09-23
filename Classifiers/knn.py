import csv
import random
import math
import operator
import numpy as np

def loadDataset(filename):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(19):
                dataset[x+1][y+1] = float(dataset[x+1][y+1])
    dataset.pop(0)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
 
# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(train_set, test_set, k, algorithm):
    scores = list()
    predicted = k_nearest_neighbors(train_set, test_set, k, algorithm)
    actual = [row[0] for row in test_set]
    accuracy = accuracy_metric(actual, predicted)
    scores.append(accuracy)
    return scores
 
# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    print(len(row1),len(row2))
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i+1] - row2[i+1])**2
    return math.sqrt(distance)

def cosine_distance(row1, row2):
    return 1-(sum([row1[i+1]*row2[i+1] for i in range(len(row1)-1)] )/(math.sqrt(sum([row1[i+1]**2 for i in range(len(row1)-1)]))*math.sqrt(sum([row2[i+1]**2 for i in range(len(row2)-1)]))))
 
# Locate the most similar neighbors

def get_neighbors(train, test_row, num_neighbors, algorithm):
    distances = list()
    for train_row in train:
        if algorithm == "euclidean":
            dist = euclidean_distance(test_row, train_row)
        elif algorithm == "cosine":
            dist = cosine_distance(test_row, train_row)
        else:
            print("Distance criteria must be choosen between \"cosine\" and \"euclidean\", Please try again")
            exit
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors
 
# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors, algorithm):
	neighbors = get_neighbors(train, test_row, num_neighbors, algorithm)
	output_values = [row[0] for row in neighbors]
	prediction = max(set(output_values), key=output_values.count)
	return prediction
 
# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors, algorithm):
	predictions = list()
	for row in test:
		output = predict_classification(train, row, num_neighbors, algorithm)
		predictions.append(output)
	return(predictions)
 
trainDataset = loadDataset("./datasets/segmentation/segmentation.Train.csv") 
testDataset = loadDataset("./datasets/segmentation/segmentation.test.csv")


# evaluate algorithm
num_neighbors = 1
algorithm = "cosine"#"euclidean"#
scores = evaluate_algorithm(trainDataset, testDataset, num_neighbors, algorithm)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

