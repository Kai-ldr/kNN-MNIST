from math import *
import numpy as np
import random


class kNeuralNetwork:
    def __init__(self, k, prints=False):
        self.k = k
        self.X_data = []
        self.y_data = []
        self.prints = prints
        
    def distance(self, p1, p2):
        total = 0
        for i in range(len(p1)):
            total += (p1[i]-p2[i])**2
        return sqrt(total)
    
    def distances(self, c_Pred):
        dists = []
        
        for i in range(len(self.y_data)):
            dists.append(self.distance(self.X_data[i], c_Pred))
        
        return dists
    
    def fit(self, X_train, y_train):
        total = 1
        for i in range(len(X_train.shape) - 1):
            total *= X_train.shape[i+1]
        self.X_data = np.reshape(X_train, (X_train.shape[0], total)).astype(float)
        
        self.y_data = y_train
        
    def predict(self, X_test, returnClosestMatch=False, returnLikelihoods=False):
        total = 1
        for i in range(len(X_test.shape) - 1):
            total *= X_test.shape[i+1]
        X_test = np.reshape(X_test, (X_test.shape[0], total)).astype(float)
        
        y_preds = []
        for i in range(len(X_test)):
            dists = self.distances(X_test[i])

            closeDists = sorted(dists)[:self.k]
            neighbors = []
            
            for y in range(len(closeDists)):
                neighbors.append(self.y_data[dists.index(closeDists[y])])
            
            
            weighted_votes = {}
    
            for neighbor, distance in zip(neighbors, closeDists):
                # Use inverse of distance as the weight (to give closer neighbors more influence)
                weight = 1 / distance
                
                if neighbor in weighted_votes:
                    weighted_votes[neighbor] += weight
                else:
                    weighted_votes[neighbor] = weight
            
            # Find the class with the highest weighted vote
            prediction = max(weighted_votes, key=weighted_votes.get)
            y_preds.append(prediction)
            
            if self.prints: print(f"Num: {i}, Prediction: {prediction}")

        if returnClosestMatch and returnLikelihoods: return y_preds, self.X_data[dists.index(closeDists[0])], weighted_votes # Closest match and likelihoods
        if returnClosestMatch: return y_preds, self.X_data[dists.index(closeDists[0])] # Closest match
        if returnLikelihoods: return y_preds, weighted_votes # Class likelihoods
        return y_preds
    
    def accuracy_score(self, y_pred, y_true):
        correct = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                correct += 1

        return correct / len(y_true)