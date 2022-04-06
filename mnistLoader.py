#Loads the Mnist Dataset

import numpy as np
from mnist import MNIST

# takes an integer and returns a list[10] with
# 0 on every place but the index given by label
# Example: 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#          0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def vectorizeLabel(label):
    output = np.zeros(10, dtype=int)
    output[label] = 1
    return output

def formatData(images, labels):
    images = np.array(images) / 255.0 # Our neural network can only process numbers between 1 and 0
    labels = list(map(vectorizeLabel, labels)) # vectorize each label so our NN can get data from it
    labels = np.vstack(labels) # formating [array(x), array(y)] -> [[x],[y]]
    return images, labels

def loadData(type = None):
    if type == None: type = "train" # set default mode to train
    data = MNIST('data-set') # Load the Mnist data set
    if type == "train":
        images, labels = data.load_training() # get training data
    elif type == "test":
        images, labels = data.load_testing() # get testing data
    else:
        print("Invalid run mode")
        exit()
    return formatData(images, labels)
    