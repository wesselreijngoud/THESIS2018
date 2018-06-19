#!/usr/bin/env python3
# Wessel Reijngoud
# Thesis 2018
# Classifier using only numerical features
import pandas as pd
import numpy as np
import csv
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB



#Run Feats individually and together, comment out files that you're not using.
filename = "LEN+PUNCT2.csv" #length and commas and dots
# filename = "LEN+COM.csv" #length and commas
# filename = "LEN+STOP.csv" #length and dots
# filename = "LEN.csv" #just length
# filename = "COM.csv" #just commas
# filename = "STOP.csv" #just dots
# filename = "PUNCT2.csv" #dots and commas


def GetFeats():
    """Loads the csv file that contains the extra features"""
    with open(filename, 'r') as fh:
        reader = csv.reader(fh)
        # skip headers
        next(reader, None)
        csv_data = []
        for row in reader:
            csv_data.append([float(var) for var in row])
    csv_data = np.asarray(csv_data)
    y = np.array(['HT'] * 9895 + ['MT'] * 9895)

    # returns the array csv_data containing all the numerical features and
    # array y containg the labels to the classsifier
    return csv_data, y




def main():
    """Loads the dataset and executes 10 fold cross validation on it with linearsvc as a classifier and 
    a simple tfidfvectorizer on bigrams as feature"""
    # load data
    x, y = GetFeats()
    # Define a 10fold cross validation as suggested by Toral
    folds = StratifiedKFold(10, shuffle=True, random_state=1)

    y_out_total = np.array([])
    y_test_total = np.array([])
    for train_index, test_index in folds.split(x, y):
        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        clf = LinearSVC()
        clf.fit(x_train, y_train)
        y_output = clf.predict(x_test)
        y_out_total = np.append(y_out_total, y_output)
        y_test_total = np.append(y_test_total, y_test)
        print(classification_report(y_test, y_output))
    print("-----------------------------")
    print('All results:')
    print(classification_report(y_test_total, y_out_total))
    print("-----------------------------")
    print("Accuracy:")
    print(accuracy_score(y_test, y_output))
    show_most_informative_features(clf)
if __name__ == '__main__':
    main()
