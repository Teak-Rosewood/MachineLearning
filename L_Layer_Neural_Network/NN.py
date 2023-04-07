import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class NLayerNetwork:
  def __init__(self, x, y, layer_dims, learning_rate = 0.01, iterations = 1000):
    self.X = x
    self.Y = y
    self.learning_rate = learning_rate
    self.iterations = iterations
    self.layer_dims = layer_dims

  def split_data(self, test_size=0.33, random_state=42):
    self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size = test_size, random_state = 42)
    self.x_train = np.array(self.x_train.T)
    self.x_test = np.array(self.x_test.T)
    self.y_train = np.array(self.y_train.T)
    self.y_test = np.array(self.y_test.T)

  def initialize_layers(self, layer_dims):
    L = len(layer_dims)
    self.parameters = {}
    for l in range(1, L):
      self.parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
      self.parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
  def softmax(self, Z):
    e_x = np.exp(Z)
    A= e_x / np.sum(np.exp(Z))  
    cache=Z
    return A,cache  

  def sigmoid(self, z):

    A = 1/(1+np.exp(-z))
    cache = z
    return A, cache

  def sigmoid_backward(self, dA, cache):
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

  def relu(self, z):

    A = np.maximum(0,z)
    cache = z 
    return A, cache

  def relu_backward(self, dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ



  def linear_forward(self, A, W, b):

    Z = W.dot(A) + b
    cache = (A, W, b)
    return Z, cache
  
  def linear_activation_forward(self, A_prev, W, b, activation):
    if activation == "sigmoid":
      Z, linear_cache = self.linear_forward(A_prev, W, b)
      A, activation_cache = self.sigmoid(Z)
    
    elif activation == "relu":
      Z, linear_cache = self.linear_forward(A_prev, W, b)
      A, activation_cache = self.relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache
  
  def forward_propogation (self, A):
    self.caches = []
    L = len(self.parameters) // 2
    for l in range (1, L):
      A_prev = A
      A, cache = self.linear_activation_forward(A_prev, self.parameters['W' + str(l)],self.parameters['b' + str(l)], 'relu')
      self.caches.append(cache)
    self.AL, cache = self.linear_activation_forward(A, self.parameters['W' + str(L)],self.parameters['b' + str(L)], 'sigmoid')
    self.caches.append(cache)  

    return self.AL, self.caches
  
  def calculate_cost (self):
    m = self.y_train.shape[1]
    cost = -(np.sum(np.multiply(self.y_train,np.log(self.AL)) + np.multiply((1-self.y_train), np.log(1-self.AL))))/m
    return cost

  def linear_backward(self, dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = dZ.dot(A_prev.T)/m
    db = np.sum(dZ, axis = 1, keepdims = True)/m
    dA_prev = W.T.dot(dZ)
    return dA_prev, dW, db

  def linear_activation_backward(self, dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
      dZ = self.relu_backward(dA, activation_cache)
      dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
      dZ = self.sigmoid_backward(dA, activation_cache)
      dA_prev, dW, db = self.linear_backward(dZ, linear_cache)        
    
    return dA_prev, dW, db

  def backward_propogation(self):
    self.grads = {}
    L = len(self.caches)
    m = self.AL.shape[1]
    Y = self.y_train.reshape(self.AL.shape)
    dAL = -(np.divide(Y, self.AL) - np.divide((1 - Y),(1 -self.AL)))

    current_cache = self.caches[L-1]
    dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(dAL, current_cache, 'sigmoid')
    self.grads["dA" + str(L-1)] = dA_prev_temp
    self.grads["dW" + str(L)] = dW_temp
    self.grads["db" + str(L)] = db_temp

    for l in reversed(range(L-1)):
      current_cache = self.caches[l]
      dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(self.grads["dA" + str(l+1)], current_cache, 'relu')
      self.grads["dA" + str(l)] = dA_prev_temp
      self.grads["dW" + str(l+1)] = dW_temp
      self.grads["db" + str(l+1)] = db_temp
  
  def update_parameters (self):
    L = len(self.parameters) // 2
    for l in range(L):
      self.parameters["W" + str(l+1)] = self.parameters["W" + str(l+1)] - self.learning_rate * self.grads["dW" + str(l+1)]
      self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - self.learning_rate * self.grads["db" + str(l+1)]

  def predict(self, X, Y):
    p = np.zeros((1, X.shape[1]))
    probs, caches = self.forward_propogation(X)
    for i in range (0, probs.shape[1]):
      if(probs[0,i] > 0.5):
        p[0,i] = 1
      else:
        p[0, i] = 0
    return np.sum((p == Y)/X.shape[1])

  def fit(self):
    costs = []
    self.initialize_layers(self.layer_dims) 
    self.split_data()
    for i in range (self.iterations):
      self.forward_propogation(self.x_train)
      cost = self.calculate_cost()
      self.backward_propogation()
      self.update_parameters()
      if i % 30000 == 0 or i == self.iterations - 1:
        print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
      if i % 30000 == 0 or i == self.iterations:
        costs.append(cost)
    print('Train accuracy:' + str(self.predict(self.x_train, self.y_train)))
    print('Test accuracy:' + str(self.predict(self.x_test, self.y_test)))
