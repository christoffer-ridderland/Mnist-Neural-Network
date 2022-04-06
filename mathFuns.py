import numpy as np

def sigmoid(z):
    return( 1/( 1 + np.exp(-z)))

def sigmoidDerivative(z):
    return( sigmoid(z) * (1 - sigmoid(z)) )

def cost(yhat, y):
    return( (y - yhat)**2 )

def costDerivative(yhat, y):
    return( 2*(y - yhat) )