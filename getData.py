import pandas as pd
import numpy as np


def getTrainData():
    # no is 0
    data = pd.read_csv("trainingset.csv", header=None)
    data = data.drop(0, axis=1)
    from sklearn.preprocessing import LabelBinarizer

    for i in range(data.shape[0]):
        try:
            data[2][i] = float(data[2][i].split('JobCat')[1])
        except:
            data[2][i] = np.nan
    for column in [3, 4, 5, 7, 8, 9, 15, 16]:
        LB = LabelBinarizer()
        data[column] = LB.fit_transform(data[column])

    Month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    for n, key in enumerate(Month):
        data[11].values[data[11] == key] = n
    np.savetxt("TrainData.csv", data.values)


def getTestData():
    data = pd.read_csv("queries.csv", header=None)
    data = data.drop(0, axis=1)
    data = data.drop(16, axis=1)
    from sklearn.preprocessing import LabelBinarizer

    for i in range(data.shape[0]):
        try:
            data[2][i] = float(data[2][i].split('JobCat')[1])
        except:
            data[2][i] = 0
    for column in [3, 4, 5, 7, 8, 9, 15]:
        LB = LabelBinarizer()
        data[column] = LB.fit_transform(data[column])

    Month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    for n, key in enumerate(Month):
        data[11].values[data[11] == key] = n
    np.savetxt("TestData.csv", data.values)


if __name__ == '__main__':
    getTrainData()
    getTestData()
