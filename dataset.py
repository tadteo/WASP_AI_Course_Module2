#!/usr/bin/env python3
import numpy as np

def generate_halfmoon(n1, n2, max_angle=np.pi):
    alpha = np.linspace(0, max_angle, n1)
    beta = np.linspace(0, max_angle, n2)
    X1 = np.vstack([np.cos(alpha), np.sin(alpha)]) + 0.1 * np.random.randn(2,n1)
    X2 = np.vstack([1 - np.cos(beta), 1 - np.sin(beta) - 0.5]) + 0.1 * np.random.randn(2,n2)
    y1, y2 = -np.ones(n1), np.ones(n2)
    return X1, y1, X2, y2

def convert_dataset(dataset):
    #convert the dataset in a shuffled numpy array (3 by length)
    
    X1 = np.array(dataset[0])
    y1 = np.array(dataset[1])

    class1 = np.append(X1,y1)
    class1 = np.reshape(class1,(3,100))
    class1 = class1.T
    
    X2 = np.array(dataset[2])
    y2 = np.array(dataset[3])

    class2 = np.append(X2,y2)
    class2 = np.reshape(class2,(3,100))
    class2 = class2.T
         
    data = np.concatenate((class1,class2))
    
    #shuffle training dataset
    np.random.shuffle(data)
    
    return data

if __name__ == "__main__":
    X1, y1, X2, y2 = generate_halfmoon(n1=100, n2=100, max_angle=2)
    # np.savez('halfmoon.npz', X1=X1, y1=y1, X2=X2, y2=y2)
    