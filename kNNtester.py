from kNN import kNeuralNetwork
import numpy as np
from multiprocessing import Pool, cpu_count

data = np.genfromtxt(fname='mnist_train.csv', delimiter=',')

data = np.delete(data, 0, 0)

data = data[0:1000, :]

allLabels = data[:, 0]

data = np.delete(data, 0, 1)

training = data[0:750, :]
testing = data[750:-1, :]
trainingLabels = allLabels[0:750]
testingLabels = allLabels[750:-1]

print("Data loaded")

kNN = kNeuralNetwork(4, prints=False)
kNN.fit(training, trainingLabels)
y_pred = kNN.predict(testing)
print(kNN.accuracy_score(y_pred, testingLabels))