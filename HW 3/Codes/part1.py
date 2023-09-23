import csv
import random
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

def load_dataset(filename):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(19):
                dataset[x+1][y+1] = float(dataset[x+1][y+1])
    dataset.pop(0)
    return dataset

def get_accuracy(actual, predicted):
	truePrediction = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			truePrediction += 1
	return truePrediction / float(len(actual)) 

def euclidean_distance_cretaria(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i+1] - row2[i+1])**2
    return math.sqrt(distance)

def cosine_distance_cretaria(row1, row2):
    return 1-(sum([row1[i+1]*row2[i+1] for i in range(len(row1)-1)] )/(math.sqrt(sum([row1[i+1]**2 for i in range(len(row1)-1)]))*math.sqrt(sum([row2[i+1]**2 for i in range(len(row2)-1)]))))

def get_k_nearest_neighbors_of_point(train, test_row, k, distanceCretaria):
    distances = list()
    for train_row in train:
        if distanceCretaria == "euclidean":
            dist = euclidean_distance_cretaria(test_row, train_row)
        elif distanceCretaria == "cosine":
            dist = cosine_distance_cretaria(test_row, train_row)
        else:
            print("Distance criteria must be choosen between \"cosine\" and \"euclidean\", Please try again")
            exit
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighborsVector = list()
    for i in range(k):
        neighborsVector.append(distances[i][0])
    return neighborsVector
 
def KNN_algorithm(train_set, test_set, k, distanceCretaria):
    predicted = classification(train_set, test_set, k, distanceCretaria)
    actual = [row[0] for row in test_set]
    accuracy = get_accuracy(actual, predicted)
    return accuracy
  
def predict_class(train, test_row, k, distanceCretaria):
	neighborsVector = get_k_nearest_neighbors_of_point(train, test_row, k, distanceCretaria)
	outputs = [row[0] for row in neighborsVector]
	prediction = max(set(outputs), key=outputs.count)
	return prediction
 
def classification(train, test, k, distanceCretaria):
	predictions = list()
	for row in test:
		output = predict_class(train, row, k, distanceCretaria)
		predictions.append(output)
	return(predictions)
 
trainDataset = load_dataset("./datasets/segmentation/segmentation.Train.csv") 
testDataset = load_dataset("./datasets/segmentation/segmentation.test.csv")

accuracyVectorForEuclidean = list()
accuracyVectorForCosine = list()

for i in range(8):
    k = i + 1
    distanceCretaria = "euclidean"
    accuracy = KNN_algorithm(trainDataset, testDataset, k, distanceCretaria)
    accuracyVectorForEuclidean.append(accuracy)
    distanceCretaria = "cosine"
    accuracy = KNN_algorithm(trainDataset, testDataset, k, distanceCretaria)
    accuracyVectorForCosine.append(accuracy)
    
plt.figure()
plt.plot([x+1 for x in range(len(accuracyVectorForEuclidean))] ,accuracyVectorForEuclidean,'-', label='KNN with euclidean distance criteria')
plt.plot([x+1 for x in range(len(accuracyVectorForCosine))] ,accuracyVectorForCosine,'-', label='KNN with Cosine similarity criteria')
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0%}'))
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Accuracy (%)")  
plt.ylim((0,1))
plt.title('K Nearest neighbors Classifier')
plt.legend()
plt.show()

print(accuracyVectorForEuclidean)
print(accuracyVectorForCosine)