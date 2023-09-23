import random
import math
import operator
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
import numpy as np
import string

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
    print(output)
    return output


classes = list()
text = list()
with open("./datasets/IMDB_review_labels.txt", encoding="utf8") as f:
    for line in f:
        temp = line.split('  \t')
        clean_data(temp[0])
        #text.append()
        #if temp[0] not in classes:
        #    classes
            

print(classes)
            
            
            
            
            
            