#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier


def LoadData():
	"""Loads the dataset that contains all the sentences"""
	x1 = []
	x2 = []
	filename = 'modifieddata2.csv'

	data = pd.read_csv(filename, delimiter=",", header=None, error_bad_lines=False, warn_bad_lines=False)

	df = pd.DataFrame(data)
	#define column names
	data.columns= ['OG','HT','MT']
	
	#makes lists for HT and MT that contains a sentence as every item in list
	for line in df['HT']:
		x1.append(line.lower().strip())
	for line in df['MT']:
		x2.append(line.lower().strip())

    # creates arrays, x with all sentences, y with all labels that are
    # returned to the classifier
	y = np.array(['HT'] * len(x1) + ['MT'] * len(x2))
	x = np.array(x1 + x2)
	

	return x, y
	
def main():
	"""Loads the dataset and runs a dummy classifier on it to create a baseline result"""
	#load data
	x, y = LoadData()
	#Define a 10fold cross validation as suggested by Toral
	folds = StratifiedKFold(10, shuffle=True, random_state=1)

	#If True use tfidf vectorizer if not use count vectorizer
	tfidf = True
	if tfidf:
		vec = TfidfVectorizer(stop_words=set(
            stopwords.words('dutch')),binary=True, ngram_range=(1, 1))
	else:
		vec = CountVectorizer(stop_words=set(
            stopwords.words('dutch')),binary=True, ngram_range=(1, 1))



	y_out_total = np.array([])
	y_test_total = np.array([])
	for train_index, test_index in folds.split(x, y):
		x_train = x[train_index]
		x_test = x[test_index]
		y_train = y[train_index]
		y_test = y[test_index]
		pipeline = Pipeline([('vec', vec), ('classifier', DummyClassifier())])
		pipeline.fit(x_train, y_train)
		y_output = pipeline.predict(x_test)
		y_out_total = np.append(y_out_total, y_output)
		y_test_total = np.append(y_test_total, y_test)
		print(classification_report(y_test, y_output))
	print("-----------------------------")
	print('All results:')
	print(classification_report(y_test_total, y_out_total))
	print("-----------------------------")
	print("Accuracy:")
	print(accuracy_score(y_test,y_output))
if __name__ == '__main__':
	main()
