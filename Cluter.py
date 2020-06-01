from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

trainData = pd.read_csv('TrainData.csv', header=None, delimiter=' ').values
trainData = np.nan_to_num(trainData)
testData = pd.read_csv('TestData.csv', header=None, delimiter=' ').values
testData = np.nan_to_num(testData)
trainX = trainData[:, 0:-1]
trainY = trainData[:, -1]
trainX, validationX, trainY, validationY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

clf = KNeighborsClassifier()
clf.fit(trainX, trainY)
PredictY = clf.predict(validationX)
print(classification_report(validationY, PredictY))

Result = clf.predict(testData).astype(np.str_)
Result[Result == '0.0'] = 'no'
Result[Result == '1.0'] = 'yes'
ID = np.loadtxt("queries.csv", delimiter=',', dtype=np.str_)
ID = np.vstack((ID[:, 0], Result))
np.savetxt("result.csv", ID.T, fmt="%s,%s")
