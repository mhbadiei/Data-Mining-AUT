import math
import numpy as np
import string
import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#from sklearn import cross_validation
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB

#reading the data from the CSV file
def create_data():
    with open("./datasets/IMDB_review_labels.txt", encoding="utf8") as f:
        text = []
        label = []
        for line in f:
            temp = line.split('  \t')
            text.append(temp[0])
            label.append(temp[1])
            
        return text, label 

    
#removing stopwords, punctuations and special characters from the description
#lemmatizing the words
def clean_data(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    stops = stopwords.words('english')
    nonan = re.compile(r'[^a-zA-Z ]')
    output = []
    for i in range(len(text)):
        sentence = nonan.sub('', text[i])
        words = word_tokenize(sentence.lower())
        filtered_words = [w for w in words if not w.isdigit() and not w in stops and not w in string.punctuation]
        tags = pos_tag(filtered_words)
        cleaned = ''
        for word, tag in tags:
          if tag == 'NN' or tag == 'NNS' or tag == 'VBZ' or tag == 'JJ' or tag == 'RB' or tag == 'NNP' or tag == 'NNPS' or tag == 'RBR':
            cleaned = cleaned + wordnet_lemmatizer.lemmatize(word) + ' '
        output.append(cleaned.strip())
    return output


#feature extraction - creating a tf-idf matrix
def calculate_tfidf(text, unique_words):
    text_list = list()
    for i in text:
        print(i)
        line = i.split(' ')
        word_list = {}
        line_list = list()
        for k in line:
            if k not in word_list:
                word_list[k]=1
            else:
                word_list[k]=word_list[k]+1
            
        for j in unique_words.keys():
            #print(j)
            if j in word_list.keys():
                line_list.append(math.log(1+word_list[j])+math.log(len(text)/unique_words[j]))
            else:
                line_list.append(math.log(len(text)/unique_words[j]))
        text_list.append(line_list)
    return text_list
    
def get_word_frequency(text):
    unique_words = {}
    for i in text:
        line = i.split(' ')
        for j in line:
            if j not in unique_words:
                unique_words[j] = 1
            else:
                unique_words[j] = unique_words[j] + 1
    return unique_words

#Naive Bayes classifier
def test_NaiveBayes(x_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    NBClassifier = gnb.fit(x_train, y_train)
    predictions = NBClassifier.predict(x_test)
    a = accuracy_score(y_test, predictions)
    p = precision_score(y_test, predictions, average = 'weighted')
    r = recall_score(y_test, predictions, average = 'weighted')
    return a, p, r


#SVM classifier
def test_SVM(x_train, x_test, y_train, y_test):
    SVM = SVC(kernel = 'linear')
    SVMClassifier = SVM.fit(x_train, y_train)
    predictions = SVMClassifier.predict(x_test)
    a = accuracy_score(y_test, predictions)
    p = precision_score(y_test, predictions, average = 'weighted')
    r = recall_score(y_test, predictions, average = 'weighted')
    return a, p, r


#Multilayer Perceptron classfier
def test_NN(x_train, x_test, y_train, y_test):
    NN = MLPClassifier(solver = 'lbfgs', alpha = 0.00095, learning_rate = 'adaptive', learning_rate_init = 0.005, max_iter = 300, random_state = 0)
    Perceptron = NN.fit(x_train, y_train)
    predictions = Perceptron.predict(x_test)
    a = accuracy_score(y_test, predictions)
    p = precision_score(y_test, predictions, average = 'weighted')
    r = recall_score(y_test, predictions, average = 'weighted')
    return p, r


#SGD classifier
#classifier that minimizes the specified loss using stochastic gradient descent
#hinge loss works pretty well too, the modified huber reports highest precision
def test_SGD(x_train, x_test, y_train, y_test):
    SGD = SGDClassifier(loss = 'modified_huber')
    SGDC = SGD.fit(x_train1, y_train)
    predictions = SGDC.predict(x_test1)
    a = accuracy_score(y_test, predictions)
    p = precision_score(y_test, predictions, average = 'weighted')
    r = recall_score(y_test, predictions, average = 'weighted')
    return p, r


#Voting Classifiers
#SVC and SGD combined with equal weights gave 1% lower precision than SVC
def test_voting(x_train, x_test, y_train, y_test):
    SVM = SVC(kernel = 'linear', probability = True)
    SGD = SGDClassifier(loss = 'modified_huber')
    EnsembleClassifier = VotingClassifier(estimators = [('sgd', SGD), ('svc', SVM)], voting = 'soft', weights = [1,1])
    EnsembleClassifier = EnsembleClassifier.fit(x_train, y_train)
    predictions = EnsembleClassifier.predict(x_test)
    a = accuracy_score(y_test, predictions)
    p = precision_score(y_test, predictions, average = 'weighted')
    r = recall_score(y_test, predictions, average = 'weighted')
    return p, r


	
	

text, label = create_data()
text = clean_data(text)
label = [x[1] for x in label[:]]

#print(text)
#print(len(text))
unique_words = get_word_frequency(text)    
#print(len(unique_words))

data = calculate_tfidf(text, unique_words)
print(len(data), len(data[1]))

#joining the title and description to create a single text document for each product
#combined = text[:]
#for i in range(len(combined)):
#    combined[i] = text[i]
#print(combined)

#feature extraction
#training = tfidf(text)
#print(training)
#training and test data splits
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.2, random_state = 0)



accuracy, precision, recall = test_NaiveBayes(x_train, x_test, y_train, y_test)
print(accuracy, precision, recall)


#gnb = GaussianNB()
#y_pred = gnb.fit(x_train, y_train).predict(x_test)
#print("Number of mislabeled points out of a total %d points : %d"% (x_test.shape[0], (y_test != y_pred).sum()))

#print(len(x_train),len(x_test),len(y_train),len(y_test))
#test a classifier
accuracy, precision, recall = test_SVM(x_train, x_test, y_train, y_test)
print(accuracy, precision, recall)
