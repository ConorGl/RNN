# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 11:22:34 2018

@author: ConorGL

We will create the class RNNTheano, which will be an RNN
"""
import numpy as np
import Functions as fn

class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        #Assign the instance variables 
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        #Randomly initialize the network parameters, they depend upon the size
        #of the vocabulary set and the size (x,y) denotes the direction of 
        #forward propogation
        self.U = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(hidden_dim, hidden_dim))
        
    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        # We start by creating a matrix of 0s in the shape of our input vector
        # + 1. This extra element will be used as the initial hidden layer
        # which will be set to 0 initially as the first input has no hidden
        # inputs yet.
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # We now create the output as a 0 matrix in the shape of the input
        # vector by the number of words we have in the vocabulary
        o = np.zeros((T, self.word_dim))
        # Finally, we work out what we want to do for each time step
        # We are using tanh as our activation function 
        for t in np.arange(T):
            s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
            o[t] = fn.softmax(self.V.dot(s[t]))
        # Both the hidden dim and current weights are output
        return [o, s]
        
    
    def predict(self, x):
        #Our prediciton is mearly a single forword propagation with our input
        o, s = self.forward_propagation(x)
        #Once the weights have all been calculated with this FP, then we only
        #need to return the index of the mostly likely word. Later this is combined
        # with the index_to_words array in order to translate this into a word
        return np.argmax(o, axis=1)
      
    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence in our input matrix x (each sentence)
        for i in np.arange(len(y)):
            # Calculate the forward_propagation matrices o, s for each setence
            o, s = self.forward_propagation(x[i])
            # Now we need to select only the parts of our o matrix
            # where the 'correct' words lay as only they are used to calculate
            # the loss
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were in our predictions
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L
 
    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x,y)/N

