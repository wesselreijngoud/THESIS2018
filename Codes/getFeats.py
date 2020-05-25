
def getFeats(x):
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
