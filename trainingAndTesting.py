from mathFuns import sigmoid, sigmoidDerivative, cost, costDerivative
from mnistLoader import loadData
from os.path import exists
import numpy as np

def initNetwork():
    # n x m matrix with values from -0.5 to 0.5
    # size of the Network is 784 (Input) -> 226 -> 128 -> 10 (output)
    # Randomize Weight maticies
    w2 = np.random.random((784, 226)) - 0.5
    w3 = np.random.random((226, 128)) - 0.5
    w4 = np.random.random((128,10)) - 0.5
    # Save the matricies
    np.save('w2.npy', w2)
    np.save('w3.npy', w3)
    np.save('w4.npy', w4)


def trainOnce(images, labels):
    if not (exists("w2.npy") and exists("w3.npy") and exists("w4.npy")):
        initNetwork()
    w2 = np.load("w2.npy")
    w3 = np.load("w3.npy")
    w4 = np.load("w4.npy")
def test(images, labels):
    w2 = np.load("w2.npy")
    w3 = np.load("w3.npy")
    w4 = np.load("w4.npy")
    right = 0
    wrong = 0
    for index, image in images:
        image1D = image.reshape(1, 784) # collapses it to one dimention
        z2 = image.dot(w2) # matrix multiplication
        a2 = sigmoid(z2)   # normalize the values with sigmoid
        z3 = a2.dot(w3)
        a3 = sigmoid(z3)
        z4 = a3.dot(w4)
        a4 = sigmoid(z4)
        
        guess = np.argmax(a4)
        answer = np.argmax(labels[index])
        if guess == answer:
            right += 1
        else:
            wrong += 1
        
        # Is the guess correct
    print("--------------------------")
    print("Wrong gusesses: ", wrong)
    print("Right gusesses: ", right)
    print("Ratio: ", right/(right+wrong))
    print("--------------------------")



def main(mode):
    if mode == 1: # TRAIN ONCE
        images, labels = loadData("train")
        trainOnce(images, labels)
    elif mode == 2: # TEST
        images, labels = loadData("test")
        test()
    elif mode == 3: # INIT AND TRAIN
        images, labels = loadData("train")
        initNetwork()
        trainOnce()
    elif mode == 4: # TRAIN X TIMES
        print("How many times?")
        count = int(input())
        images, labels = loadData("train")
        for _ in range(count-1): trainOnce()
        
    else:
        print("No valid run mode presented: exiting")
        exit()
        


def run():
    print("1: Train existing network on the data once")
    print("2: Test Neural Network")
    print("3: Initialize and train neural network")
    print("WARNING: Initializing will erase the network's previous training")
    print("4: Train x times")
    mode = int(input())
    print(mode)
    if mode < 0 or mode > 4:
        print("invalid mode")
        exit()
    main(mode)
    
run()