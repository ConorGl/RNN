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
    
    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # Create the variables to hold the gradients in for SGD
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # We then calculate the gradients, using the most recent output first
        # then back-prop through time
        for t in np.arange(T)[::-1]:
          dLdV += np.outer(delta_o[t], s[t].T)
          # Initial delta calculation
          delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
          # Backpropagation through time (for at most self.bptt_truncate steps)
          for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
              # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
              dLdW += np.outer(delta_t, s[bptt_step-1])              
              dLdU[:,x[bptt_step]] += delta_t
              # Update delta for next step
              delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]
    
    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print("Performing gradient check for parameter {} with size {}.".format(pname, np.prod(parameter.shape)))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error &gt; error_threshold:
                    print("Gradient Check ERROR: parameter={} ix={}".format(pname, ix))
                    print("+h Loss: {}".format(gradplus))
                    print("-h Loss: {}".format(gradminus))
                    print("Estimated_gra1dient: {}".format(estimated_gradient))
                    print("Backpropagation gradient: {}".format(backprop_gradient))
                    print("Relative Error: {}".format(relative_error))
                    return
                it.iternext()
            print("Gradient check for parameter {} passed.".format(pname))
        
model = RNNNumpy(8000)
