# THESIS2018
Bachelor Thesis Project

All the files needed to reproduce the same results as presented in the thesis paper are included in this github. 

The two .csv files contain the following:
-LEN+PUNCT2.csv contains the preprocessed features containing sentence length difference, amount of commas difference and amount of full stops (dots '.') difference
-modifieddata2.csv contains all the source and translated sentences.

The python programs included do the following:
-dummy.py is a baseline classifier
-ClassifierFeats.py is a SVC classifier that only uses the numerical features included in LEN+PUNCT2.csv to produce results.
-ClassifierBOW.py is a SVC classifier that only uses the bag of words produced from modifieddata2.csv to produce results.
-ClassifierCombined.py is a SVC classifier that combines the two programs above to produce results
-wordcount.py is a program that counts the words and unique words in the three columns of modifieddata2.csv
-featureextract.py is a program that was written to produce the features contained in LEN+PUNCT2.csv
