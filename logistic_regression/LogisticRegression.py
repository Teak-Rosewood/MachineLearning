import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, x, y, learning_rate = 0.01, iterations = 1000):
        self.X = x
        self.Y = y
        self.learning_rate = learning_rate
        self.iterations = iterations
    
    def initialize_weights(self, dim):
        self.w = np.zeros((dim, 1))
        self.b = np.float64(0)
    
    def get_weights(self):
        return self.w, self.b

    def split_data(self, test_size=0.33, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size = test_size, random_state = 42)
        self.x_train = np.array(self.x_train.T)
        self.x_test = np.array(self.x_test.T)
        self.y_train = np.array(self.y_train.T)
        self.y_test = np.array(self.y_test.T)
        self.m = self.x_train.shape[1]
    
    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z))
        return s
    
    def calculate_cost(self, A):
        cost = -(np.sum((self.y_train * np.log(A)) + ((1 - self.y_train) * np.log(1 - A)))) / self.m
        return cost
    
    def forward_propogation(self):
        A = self.sigmoid(np.dot(self.w.T, self.x_train) + self.b)
        cost = self.calculate_cost(A)
        cost = np.squeeze(np.array(cost))
        return A, cost
    
    def backward_propogation(self, A):
        dw = np.dot(self.x_train, (A-self.y_train).T) / self.m
        db = np.sum(A-self.y_train) / self.m
        grads = {"dw": dw,
                 "db": db}
        return grads
    
    def predict(self, X):
        Y_prediction = np.zeros((1, X.shape[1]))
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)
        for i in range(A.shape[1]):
            if A[0, i] > 0.5:
                Y_prediction[0,i] = 1
            else:
                Y_prediction[0,i] = 0
        return Y_prediction
    
    def optimize(self):
        for i in range(self.iterations):
            A, cost = self.forward_propogation()
            grads = self.backward_propogation(A)
            dw = grads["dw"]
            db = grads["db"]
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
            if i % 100000 == 0:
                self.costs.append(cost)
                print ("Cost after iteration %i: %f" %(i, cost))
        
    def fit(self):
        # initializing variables and splitting data set
        
        self.split_data()
        self.initialize_weights(self.x_train.shape[0])
        self.costs = []
        
        # Applying gradient descent

        self.optimize()
        
        # Printing test/train accuracy
        
        self.Y_prediction_train = self.predict(self.x_train)
        train_score = 100 - np.mean(np.abs(self.Y_prediction_train - self.y_train)) * 100
        
        self.Y_prediction_test = self.predict(self.x_test)
        print(self.x_test.shape, self.y_test.shape)
        test_score = 100 - np.mean(np.abs(self.Y_prediction_test - self.y_test)) * 100
        
        print("Train accuracy: {} %".format(train_score))
        print("Test accuracy: {} %".format(test_score))