# Data Mining Exercises and Solutions Repository

Welcome to the Data Mining Exercises repository! This repository contains a collection of exercises and questions related to data mining and machine learning concepts. These exercises are designed to help you understand and practice various aspects of data mining, from fundamental concepts to practical implementation.

## Contents

- **Exercise 1:** This section covers fundamental concepts in data mining, including supervised and unsupervised learning, evaluation metrics, data preprocessing, association rules mining, and more. Refer to the `Exercise1` folder.

- **Exercise 2:** Exercise 2 delves into practical applications of clustering and classification algorithms, exploring theoretical questions and practical implementations using datasets such as Iris and Worms. Refer to the `Exercise2` folder.

- **Exercise 3:** The third exercise set focuses on theoretical and practical aspects of classification algorithms, including Naïve Bayes, Decision Trees, Support Vector Machines, and k-Nearest Neighbors. Refer to the `Exercise3` folder.

## How to Use

1. Clone this repository to your local machine using `git clone`.

2. Navigate to the specific exercise folder you want to work on (e.g., `Exercise1`, `Exercise2`, or `Exercise3`).

3. Read the questions provided in each exercise's pdf file.

4. Implement your solutions to the exercises, and add your code to the respective folders.

5. For figures and diagrams related to the exercises, please refer to the PDF files included in each exercise folder.

6. Feel free to use the provided answers as references, but aim to complete the exercises on your own first.

7. If you have questions or need further clarification, please open an issue in this repository.

## Prerequisites

- Basic knowledge of data mining and machine learning concepts.

- Familiarity with programming in Python and relevant libraries such as scikit-learn and pandas.

# Exercise 1
Question 1
Define the following concepts comprehensively:

Unsupervised Learning
Supervised Learning
Semi-supervised Learning
Outlier
Dimension
Training, Validating, and Testing Data
Data Warehousing
Missing Values
Independent Variable
Question 2
Dimension reduction is a common technique in data mining to enhance data efficiency. Many features do not have significant impact, and their presence or absence is not crucial. Therefore, using techniques, it is possible to identify features that can be removed first and then extract important features from them. Mention at least two of these techniques, explain one of them with an example, and describe the difference between feature selection and feature extraction.

Question 3
Introduce precision (P), recall (R), and F-score (F) evaluation metrics based on the confusion matrix (C).

Question 4
Assume that the correlation between two variables is zero. What is the concept? Based on the definition in question 1, are these variables independent of each other?

Question 5
Data preprocessing is an essential step in data-driven projects. In natural language processing, preprocessing involves tasks like removing stop words, eliminating spaces, and more. However, in data mining, the focus is primarily on numerical data. Various techniques such as data cleaning, data integration, and data transformation can be mentioned. Explain these three mentioned techniques.

Question 6
Execute the Apriori algorithm on the following transactions. Assume a support threshold of 33% and a confidence threshold of 60%. Show all the stages of candidate itemset generation and finally obtain the frequent itemsets. Also, list all the association rules that can be generated from these frequent itemsets, identify those that are certain, and order them by confidence.

Question 7
a. Using the transactions from the previous question and with the same support threshold, create a frequent pattern tree. Show how the tree grows with each transaction.

b. Using the FP-Growth algorithm, find the frequent itemsets in this tree.

Section: Data Preprocessing

Preprocessing

The goal of this section is to become familiar with data preprocessing techniques and the use of data. Generally, two prominent libraries are used in this section:

Pandas
scikit-learn
First, learn the commands in these libraries and then proceed to the next section. FIFA 2021, which was released a few months ago, is one of the exciting console games. However, in this section, we are only going to use player information from the game, not the game itself!

The attached dataset in this exercise is the player information dataset of this game. Now, apply the following tasks to the dataset:

Read the dataset 'players.csv' and display both the beginning and the end of it.
Find missing values according to the definition section.
Calculate the average, maximum, and minimum weight of players.
Which country has the most and least players? Report the number of players for each country.
Find and report the most promising players with Growth > 4 and Potential > 84.
Report the positions of these promising players in the game.
Which club has the most promising players? How many players?
What is the total value of promising players in Chelsea Football Club?
How many players in 2021 have their contracts ended with their clubs and are not playing in their national teams? Report it.
Report the position, income, and current club of Mehdi Taremi.
Section: Association Rules Mining

Association Rules Mining

In this section, we aim to first get familiar with the Weka software tool. Then, we will learn how to use the Weka library.

Follow the commands in the 'weka_guide.pdf' file and answer the question at the end of it.

Use 'AssociationRulesMining.java' as a template for the following tasks. You can also refer to the codes available in the 'Examples' folder.

Using the Weka library, write Java programs that load the supermarket dataset and compare the runtime of two algorithms for different parameter values, Supmin = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15], Confmin = 0.9, and Number of Rules = 100000, by plotting their execution times.

(Optional) Write Java programs using the Weka library that load the supermarket dataset and, for the given parameter values, use the Growth-FP algorithm to first generate all frequent itemsets (except length 1) in a two-step process. Then, output all closed itemsets and maximal itemsets in a text file. Exclude frequent itemsets of length 1.


# Exercise 2
Part One: Theoretical Questions

We want to cluster N data points using k-means clustering into three clusters. Consider the objective function for clustering:

a. Execute the k-means clustering algorithm and find the cluster centers for each of the three clusters. (Assume data points 3A, 4A, and 8A belong to the first cluster, 2A, 5A, and 7A belong to the second cluster, and 1A and 6A are in the third cluster.)

b. Execute the PAM (Partitioning Around Medoids) algorithm for the same dataset and find the cluster centers for each of the three clusters. When necessary, consider choosing medoids from the top to the bottom, i.e., the first medoid randomly selected (which will become the medoid for cluster 1A), the second randomly selected medoid (which will become the medoid for 2A), and so on.

Explain how to determine the optimal number of clusters (k) when using the k-means clustering algorithm.

For each of the following clustering algorithms, provide a brief explanation and mention their pros and cons:

a. k-medoids
b. CLARA
c. DBSCAN
d. OPTICS
e. BIRCH
f. CHAMELEON

Which statements about the DBSCAN clustering algorithm are correct? Explain your answers for each statement:

a. To be considered as part of a cluster, data points must be "core points" (i.e., have a sufficient number of neighbors).
b. The DBSCAN algorithm is sensitive to outliers.
c. The computational complexity of the DBSCAN algorithm is O(n^3).
d. The DBSCAN algorithm requires prior knowledge of the number of clusters.

In cases where our data is highly imbalanced in terms of cluster sizes, which method, between k-means (assuming we know the number of clusters) and DBSCAN, should we choose for clustering, and why?

Five data points with pairwise distances are mentioned in the question. Cluster the data points twice using hierarchical clustering: once with single-linkage and once with complete-linkage. Plot the dendrogram trees of the resulting clusters. (Assume that the dendrogram trees are structured properly and have consistent heights.)

Part Two: Practical Questions

In this section, we want to apply the k-means and DBSCAN clustering algorithms to two datasets: Iris and Worms. The Iris dataset contains information about iris flowers, while the Worms dataset consists of microscopes images of synthetic worms. Both datasets are available in their respective directories.

Read the Iris dataset and visualize it. You may need to reduce the dimensions for better visualization. Ensure that the plot distinguishes the different classes using colors.

Implement a function that performs k-means clustering (you can use a library) with a user-defined value of k and returns the cluster assignments. Also, implement a function to visualize the results by coloring data points according to their assigned clusters.

Apply the k-means clustering function to the Iris dataset for k values of 1, 2, 3, 4, and 5. For each k, compute the mean absolute error (MAE) of data points to their corresponding cluster centroids.

Use the "elbow" method to explain which k value seems most appropriate for clustering the Iris dataset.

Read the Worms dataset and visualize it. You may need to reduce the dimensions for better visualization. Ensure that the plot distinguishes the different classes using colors.

Perform k-means clustering on the Worms dataset with a reasonable value of k, and visualize the results as done for the Iris dataset.

Apply the DBSCAN algorithm to the Worms dataset. Tune the epsilon (ε) and minimum samples (MinPts) parameters to obtain meaningful clusters. Visualize the results.

In this section, we aim to cluster the Worms dataset using the DBSCAN algorithm and identify the number of clusters based on the output. Explain the parameters epsilon (ε) and minimum samples (MinPts) by adjusting them such that the output provides a sensible number of clusters. Ensure that the reported number of clusters is reasonable, even though there may be no clear answer.

# Exercise 3
Part One: Theoretical Questions

Suppose there is a spam detection software that automatically identifies the type of an email (spam or not) based on the presence or absence of certain words in the text. Below are the training data for this software:

a. If this software assumes a value of p(spam) = 0.1, explain whether this choice is logical or not. Why?

b. Using Bayes' rule and p(spam) = 0.1, determine the classification of the sentence "money for psychology study."

In the figure below, data points of two classes are plotted on one of the continuous variables. If we want to transform this continuous feature into a binary feature for a decision tree, which of the points 1 or 2 is more suitable for this transformation?

The table below shows the conditions of tennis matches when either Rafael Nadal or Roger Federer won.

a. Draw a decision tree for this data and calculate the information gain for each feature at each step. (F = Federer's win, N = Nadal's win)

b. Consider the following samples as a validation set and calculate the accuracy of the decision tree on these samples.

What is boosting, and how does it increase accuracy? One of the methods using the boosting idea is "boosting gradient," which results in an ensemble of decision trees to solve regression problems. Research this method and provide a brief explanation in a paragraph.

Compare the two tree pruning methods, post-pruning and pre-pruning, and mention their advantages and disadvantages.

Why do we generally normalize data?

Does ID3 guarantee reaching a solution that is globally optimal? Explain.

Show that accuracy is a function of precision and recall.

Part Two: Practical Questions
Part Two: Practical Questions

In this question, we are going to implement the KNN algorithm and use it for classifying a dataset called "segmentation." This dataset contains summary information about pixel data from images, such as the average levels of red, green, or blue, color concentration, and more. Based on these features, images are divided into 7 different categories like grass, window, sky, etc. In this dataset, the first column is the label for each data point, followed by 19 attributes for each data sample. Initially, you need to read all this data, and then the training will be done on the "segmentation.Train" dataset, and testing on the "segmentation.Test" dataset.

The KNN algorithm is not computationally expensive, making it suitable for small datasets. However, as this exercise will show, it can perform well even with small training data.

In the prediction phase, you first determine the parameter 'k' with the help of the user. Then, for each data point that you want to predict, you calculate its distances to all data points in the training set. Afterward, you choose the 'k' nearest neighbors and examine their labels. The label with the highest count among the neighbors is considered the prediction. In case of a tie (when multiple labels have the same count), you can randomly choose one.

You'll need to write a function that takes four parameters: the training set, the testing set, 'k', and the type of distance metric (either Euclidean or Cosine). It should return a list of predicted outputs for the testing set. After writing this function, for the given training and testing sets, for 'k' values [1, ..., 8], and both distance metrics (Euclidean and Cosine), calculate the accuracy (percentage of correct predictions). Finally, visualize the accuracy for each 'k' value in a graph. Identify the best combination of distance metric and 'k'.

To calculate the Cosine similarity, you'll need to normalize the vectors and then calculate their dot product.

In this part, we want to implement the Decision Tree algorithm on the "car.data" dataset provided. This dataset contains information about cars, including buying cost, maintenance cost, number of doors, passenger capacity, trunk size, and overall safety rating. The goal is to classify cars into one of four categories: unacceptable, acceptable, good, or excellent. Read the dataset, shuffle it, and split it into training and testing sets with an 80:20 ratio.

You can use the scikit-learn library for this task. Experiment with at least three different tree depths and report the accuracy for each. For the best tree depth, evaluate the model using a confusion matrix and report training and testing errors. Visualize the best decision tree obtained.

Naïve Bayes is widely used for text categorization. In this part, we aim to classify opinions about IMDB films in two categories: negative and positive, using the Naïve Bayes algorithm. Preprocess the dataset by removing punctuation, tokenizing, removing stop words, handling HTML tags, stemming, and other necessary preprocessing steps. Create a TF-IDF matrix for the text data. TF-IDF (Term Frequency-Inverse Document Frequency) measures the importance of words in documents relative to a collection of documents.

Utilize the scikit-learn library to implement Gaussian Naïve Bayes. Train the model on the TF-IDF data and report the training and testing errors.

In the final part, we aim to perform binary classification using Support Vector Machines (SVM). Similar to the previous question, you don't need to implement the SVM algorithm from scratch; you can use the scikit-learn library. Split the dataset into training and testing sets with a 90:10 ratio. Use the linear kernel for the SVM model. Train the model and report its accuracy on the testing data. Additionally, extract and visualize the support vectors and decision boundaries.

