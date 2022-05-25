#!/usr/bin/env python3

from calendar import EPOCH
from re import T
from unittest import result
from dataset import generate_halfmoon
import numpy as np
import matplotlib.pyplot as plt

EPOCH = 3
ALPHA = 0.01

def plot(training_dataset, test_dataset):
    #plot the data
    fig, ax = plt.subplots()
    color_train = 'blue'
    color_test = 'red'
    
    #training dataset to numpy array
    X1_train = np.array(training_dataset[0])
    y1_train = np.array(training_dataset[1])
    X2_train = np.array(training_dataset[2])
    y2_train = np.array(training_dataset[3])
    
    X1_test = np.array(test_dataset[0])
    y1_test = np.array(test_dataset[1])
    X2_test = np.array(test_dataset[2])
    y2_test = np.array(test_dataset[3])

    ax.scatter(X1_train[0,:], X1_train[1,:], marker='o', c=color_train, label="class 1 train", s=40, cmap=plt.cm.Spectral)
    ax.scatter(X2_train[0,:], X2_train[1,:], marker='x', c=color_train, label="class 2 train", s=40, cmap=plt.cm.Spectral)

    ax.scatter(X1_test[0,:], X1_test[1,:], marker='o', c=color_test, label="class 1 test", s=40, cmap=plt.cm.Spectral)
    ax.scatter(X2_test[0,:], X2_test[1,:], marker='x', c=color_test, label="class 2 test", s=40, cmap=plt.cm.Spectral)

    ax.legend(loc='upper right')

    plt.show()
    
def calculate_error(data,results):
    #calculate the error
    #...
    pass

def main():
    training_dataset = generate_halfmoon(n1=100, n2=100, max_angle=2)
    test_dataset = generate_halfmoon(n1=100, n2=100, max_angle=2)
    
    
    
    w = np.array([0,0])
    b = 0
    
    X1 = np.array(training_dataset[0])
    y1 = np.array(training_dataset[1])
    # print(y1.shape)
    # print(X1.shape)
    class1 = np.append(X1,y1)
    # print(class1.shape)
    class1 = np.reshape(class1,(3,100))
    class1 = class1.T
    # print(class1)
    
    X2 = np.array(training_dataset[2])
    y2 = np.array(training_dataset[3])
    # print(y2.shape)
    # print(X2.shape)
    class2 = np.append(X2,y2)
    # print(class2.shape)
    class2 = np.reshape(class2,(3,100))
    class2 = class2.T
    # print(class2)
    
    print(y1,y2)
     
    train_data = np.concatenate((class1,class2))
    print(train_data,train_data.shape)
    
    i =0
    while i<10:
        for e in range(EPOCH):
            pass          
            # F = calculate_error(training_dataset,results)
            #train the model
            #...
            #test the model
            #...
        i+=1    
    
    plot(training_dataset,test_dataset)
    pass

if __name__ == "__main__":
    main()
