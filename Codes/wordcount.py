# Wessel Reijngoud
# Thesis 2018
# Counts the number of words and unique words for the data

import pandas as pd
import re


def main():
    """Program to count words and unique words in each of the data columns"""
    HTlist = []
    MTlist = []
    OGlist = []

    # define dataset filename
    filename = 'modifieddata2.csv'
    # read csv
    data = pd.read_csv(filename, delimiter=",", header=None,
                       error_bad_lines=False, warn_bad_lines=False)
    # create dataframe
    df = pd.DataFrame(data)
    # define column names
    data.columns = ['OG', 'HT', 'MT']

    """Calculates (unique)words for the human translated Dutch data"""
    for line in df['HT']:
        line = line.split(' ')
        for line in line:
            """Strips any non alphanumeric sign"""
            line = re.sub(r'\W+', '', line)
            line = line.lower()
            HTlist.append(line)

    """Calculates (unique)words for the machine translated Dutch data"""
    for line2 in df['MT']:
        line2 = line2.split(' ')
        for line2 in line2:
            line2 = re.sub(r'\W+', '', line2)
            line2 = line2.lower()
            MTlist.append(line2)

    """Calculates (unique)words for the original English data"""
    for line3 in df['OG']:
        line3 = line3.split(' ')
        for line3 in line3:
            line3 = re.sub(r'\W+', '', line3)
            line3 = line3.lower()
            OGlist.append(line3)

    """Prints all the outcomes"""
    print("Original words:", len(OGlist))
    print("Original unique:", len(set(OGlist)))

    print("Human words:", len(HTlist))
    print("Human unique:", len(set(HTlist)))

    print("Machine words:", len(MTlist))
    print("Machine unique:", len(set(MTlist)))


main()
