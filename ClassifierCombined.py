# Wessel Reijngoud
# Thesis 2018
# Classifier with bag of words and numerical features combined

import csv
import numpy as np
from numpy import genfromtxt
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from collections import Counter


def LoadData():
    """Loads the dataset that contains all the sentences"""
    x1 = []
    x2 = []

    filename = 'modifieddata2.csv'
    data = pd.read_csv(filename, delimiter=",", header=None,
                       error_bad_lines=False, warn_bad_lines=False)
    df = pd.DataFrame(data)
    # define column names
    data.columns = ['OG', 'HT', 'MT']

    # makes lists for HT and MT that contains a sentence as every item in list
    for line in df['HT']:
        x1.append(line.lower().strip())

    for line in df['MT']:
        x2.append(line.lower().strip())

    # only include words in the vocabuary which appear more than once
    n = 1
    c = Counter(word for x in x1 for word in x.split())
    x1 = [' '.join(y for y in x.split() if c[y] > n) for x in x1]
    c2 = Counter(word for x in x2 for word in x.split())
    x2 = [' '.join(y for y in x.split() if c2[y] > n) for x in x2]

    # creates arrays, x with all sentences, y with all labels that are returned to the classifier
    y = np.array(['HT'] * len(x1) + ['MT'] * len(x2))
    x = np.array(x1 + x2)

    return x, y


def GetFeats():
    """Loads the csv file that contains the extra features"""
    with open('LEN+PUNCT2.csv', 'r') as fh:
        reader = csv.reader(fh)
        # skip headers
        next(reader, None)
        csv_data = []
        for row in reader:
            csv_data.append([float(var) for var in row])
    csv_data = np.asarray(csv_data)
    return csv_data


def show_most_informative_features(vectorizer, clf, n=10):
    """function to print most informative features, obtained but edited from https://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers"""
    listHT = []
    listMT = []
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        listHT.append(fn_1)
        listMT.append(fn_2)
    print('HT',listHT)
    print("--------")
    print('MT', listMT)

def main():
    """Creates and runs the classifier SVC with 10-fold cross validation"""

    # loads data and features
    LoadData()
    x, y = LoadData()
    csv_data = GetFeats()

    # define 10 fold cross validation
    folds = StratifiedKFold(10, shuffle=True, random_state=1)

    # If True use tfidf vectorizer if not use count vectorizer
    tfidf = True
    if tfidf:
        vec = TfidfVectorizer(stop_words=set(
            stopwords.words('dutch')), binary=True, ngram_range=(1, 1))
    else:
        vec = CountVectorizer(stop_words=set(
            stopwords.words('dutch')), binary=True, ngram_range=(1, 1))

    # vectorizer the sentences
    countvec = vec.fit(x)
    x = countvec.transform(x)
    # print(x.shape)
    y_out_total = np.array([])
    y_test_total = np.array([])
    for train_index, test_index in folds.split(x, y):
        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        # split the numerical features csv file
        csv_train = csv_data[train_index]
        csv_test = csv_data[test_index]
        # combine the numerical features and bag of words
        x_train = np.column_stack((csv_train, x_train.toarray()))
        x_test = np.column_stack((csv_test, x_test.toarray()))

        # define classifier and run classifier
        clf = LinearSVC()
        clf.fit(x_train, y_train)
        y_output = clf.predict(x_test)
        y_out_total = np.append(y_out_total, y_output)
        y_test_total = np.append(y_test_total, y_test)
        print(classification_report(y_test, y_output))
    # prints results
    print("-----------------------------")
    print('All results:')
    print(classification_report(y_test_total, y_out_total))
    print("-----------------------------")
    print("Accuracy:")
    print(accuracy_score(y_test, y_output))
    show_most_informative_features(countvec, clf)

if __name__ == '__main__':
    main()
