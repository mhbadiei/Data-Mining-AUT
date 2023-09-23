import math
import numpy as np
import string
import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def load_dataset():
    with open("./datasets/IMDB_review_labels.txt", encoding="utf8") as f:
        text = []
        label = []
        for line in f:
            temp = line.split('  \t')
            text.append(temp[0])
            label.append(temp[1])
            
        return text, label 

def get_accuracy(actual, predicted):
	truePrediction = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			truePrediction += 1
	return 100 * truePrediction / float(len(actual))

def get_precision(actual, predicted):
	truePositive = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i] and actual[i]=='1':
			truePositive += 1
	return 100 * truePositive / float(actual.count('1'))

def data_cleaning(text):
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

def calculate_tfidf(text, unique_words):
    text_list = list()
    for i in text:
        line = i.split(' ')
        word_list = {}
        line_list = list()
        for k in line:
            if k not in word_list:
                word_list[k]=1
            else:
                word_list[k]=word_list[k]+1
            
        for j in unique_words.keys():
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

text, label = load_dataset()
text = data_cleaning(text)
label = [x[1] for x in label[:]]
unique_words = get_word_frequency(text)    
data = calculate_tfidf(text, unique_words)
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.2, random_state = 0)

accuracy = list()
precision = list()


model = SVC(kernel = 'linear').fit(x_train, y_train)
pred = model.predict(x_test)
accuracy.append(get_accuracy(y_test, pred))
precision.append(get_precision(y_test, pred))

print("Accuracy of SVM Bayes classifier for the test data is ", accuracy[0])
print("Precision of SVM classifier for the test is ", precision[0], "\n")

pred = model.predict(x_train)
accuracy.append(get_accuracy(y_train, pred))
precision.append(get_precision(y_train, pred))

print("Accuracy of SVM classifier for the train datas is ", accuracy[1])
print("Precision of SVM classifier for the train datas is ", precision[1], "\n")

plot_list = [accuracy[0], precision[0], accuracy[1], precision[1]]

plt.figure(figsize=(8,8))
colors_list = ['Red','Orange', 'Blue', 'Purple']
graph = plt.bar(['Accuracy of Test','Precision of Test','Accuracy of Train','Precision of Train'], plot_list, color = colors_list)
plt.title('Accuracy and Precision of SVM Classifier')

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
plt.show()