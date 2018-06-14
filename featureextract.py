#Wessel Reijngoud
#Thesis 2018
#Features for sentence length difference and for sentence splits difference 

import csv
from sklearn.feature_extraction.text import CountVectorizer
import string
import re

#load data
filename = 'modifieddata2.csv'


def SentenceLength(f):
	"""Calculates the sentencelength difference of Human-Original and Machine-Original"""
	counter=0
	with open(filename) as f:
		read = csv.reader(f)
		for row in read:
			#Original
			zin0=row[0].split()
			#Human Translation
			zin1=row[1].split()
			#Machine Translation
			zin2=row[2].split()
			counter+=1
			#PRINT LENGTH DIFFERENCE
			#print("HT",counter,(abs(len(zin0)- len(zin1))))
			print("MT",counter,(abs(len(zin0)- len(zin2))))

def SentenceSplitsStops(f):
	"""Calculates the full stops difference of Human-Original and Machine-Original"""
	counter=0
	with open(filename) as f:
		read = csv.reader(f)
		for row in read:
			#Original
			zin0=row[0]
			#Human Translation
			zin1=row[1]
			#Machine Translation
			zin2=row[2]
			counter+=1
			#FULL STOPS
			#print(abs((zin0.count('.') - zin1.count('.'))))
			print(abs((zin0.count('.') - zin2.count('.'))))

def SentenceSplitsCommas(f):
	"""Calculates the commas difference of Human-Original and Machine-Original"""
	counter=0
	with open(filename) as f:
		read = csv.reader(f)
		for row in read:
			#Original
			zin0=row[0]
			#Human Translation
			zin1=row[1]
			#Machine Translation
			zin2=row[2]
			counter+=1
			#COMMAS 
			#print((abs(zin0.count(',') - zin1.count(','))))
			print((abs(zin0.count(',') - zin2.count(','))))


def main():
	SentenceLength(filename)	
	SentenceSplitsStops(filename)
	SentenceSplitsCommas(filename)

main()
