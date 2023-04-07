import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LinearRegression:
    def __init__(self, x, y):
        self.w = np.zeros((x.shape[1], 1))
        self.b = float(0)
        self.X = x
        self.Y = y

    def split_data(self, train_size = 0.75, rand_state = 42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, train_size=train_size, random_state= rand_state) 
        self.x_train = (np.array(self.x_train))
        self.x_test = (np.array(self.x_test))
        self.y_test = (np.array(self.y_test))
        self.y_train = (np.array(self.y_train))
        self.m = self.x_train.shape[0]

    def get_weights(self):
        return self.w, self.b
    
    def gradient_propogator(self):
        self.A = self.x_train.dot(self.w) + self.b
        
        self.cost = np.sum((self.A - self.y_train) ** 2) / (2 * self.m)
        self.db = np.sum(self.A - self.y_train)/self.m
        self.dw = self.x_train.T.dot(self.A - self.y_train)/self.m
        
    def gradient_optimizer(self, learning_rate = 0.001, epochs = 1000):
        for self.i in range(epochs):
            self.gradient_propogator()
            self.w = self.w - (learning_rate * self.dw)
            self.b = self.b - (learning_rate * self.db)
            if(self.i % 10000 == 0):
                print("Cost after %i epochs: %f" %(self.i, self.cost))
    def predict(self, X):
        output = X.dot(self.w) + self.b
        return output
