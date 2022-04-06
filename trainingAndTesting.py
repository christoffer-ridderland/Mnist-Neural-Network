from mathFuns import sigmoid, sigmoidDerivative, cost, costDerivative
from mnistLoader import loadData


def main(mode):
    if mode == 'train':
        images, labels = loadData("train")
    elif mode == 'test':
        images, labels = loadData("test")
    elif mode == 'trainFromScratch':
        trainImages, trainLabels = loadData("train")
        testImages, testLabels = loadData("test") 
    elif mode == 'trainForTime':
        images, labels = loadData("train")
    else:
        print("No valid run mode presented: exiting")
        exit()
        


def run():
    print("1: Train existing network")
    print("2: Test Neural Network")
    print("3: Initialize and train neural network")
    print("WARNING: Initializing will erase the network's previous training")
    print("4: Train for x minutes")
    mode = int(input())
    if mode < 0 or mode > 4:
        print("invalid mode")
        exit()
    main(mode)