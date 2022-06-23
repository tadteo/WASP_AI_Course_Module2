#!/usr/bin/env python3

from cProfile import label
import math

from matplotlib import animation
from dataset import convert_dataset, generate_halfmoon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


EPOCH = 3
ALPHA = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
# ALPHA = [0.02]

def plot(training_dataset, test_dataset,w,b):
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

    x = np.arange(-2, 2, 0.01)
    y = (w[0]/w[1])*x + b/w[1]
    print(x.shape,y.shape)
    ax.plot(x, y, color='black', label='Decision Boundary')
    
    ax.legend(loc='upper right')

    plt.show()
    
def calculate_error(data,result):
    x_1,x_2,y = data
    err = math.log(1+math.exp(-y*(result)))
    return err

def avg_err(dataset,w,b):
    
    err = []
    for d in range(len(dataset)):
        x_1,x_2,y = dataset[d]
        prediction = w[0]*x_1 + w[1]*x_2 + b
        err.append(calculate_error(dataset[d],prediction))
    
    
    
    return sum(err)/len(err)

def gradient_descent(x,y,w,b,alpha):
    #calculate the gradient
    
    part = math.exp(-y*(w[0]*x[0] + w[1]*x[1] + b))
    denominator = 1 + math.exp(-y*(w[0]*x[0] + w[1]*x[1] + b))
    
    F_dw = (-y*x*part)/denominator
    F_db = (-y*part/denominator)
    
    #update the model
    
    w = w - alpha*F_dw
    b = b - alpha*F_db
    return w,b
    
def main():
    training_dataset = generate_halfmoon(n1=100, n2=100, max_angle=2)
    test_dataset = generate_halfmoon(n1=100, n2=100, max_angle=2)
        
    train_data = convert_dataset(training_dataset)
    test_data = convert_dataset(test_dataset)
    # print(train_data,train_data.shape)
    
    #Loop for testing different learning rates
    for alpha in ALPHA:
        
        w = np.array([0,0])
        b = 0
        w_history = []
        b_history = []
        
        train_err = []
        test_err = []
        i_for_plot = []
        #loop to iterate multiple times trough the dataset
        for k in range(3):
            i =0
            
            #dataset loop
            while i<(200/EPOCH):
                #epoch loop
                for e in range(EPOCH):
                    
                    x_1,x_2,y = train_data[i+e]
                    # print(x_1,x_2,y)
                    
                    prediction = w[0]*x_1 + w[1]*x_2 + b
                    
                    F = calculate_error(train_data[i+e],prediction)
                    
                    x = np.array([x_1,x_2])
                    w,b = gradient_descent(x,y,w,b,alpha)
                    # print(w,b)    
                
                #calculating training error
                train_err.append(avg_err(train_data,w,b))
                # print(f"{i} - training error: {train_err}")
                #calculating test error
                test_err.append(avg_err(test_data,w,b))            
                # print(f"{i} - test error: {test_err}")
                # plot(training_dataset,test_dataset,w,b)
                w_history.append(w)
                b_history.append(b)
                i_for_plot.append(k*200+i*EPOCH)
                i+=1
            
            # plot(training_dataset,test_dataset,w,b)
            
           
            fig,ax = plt.subplots()
            
            def animate(j):
                # print(i)
                ax.clear()
                ax.set_xlim(-2,2)
                ax.set_ylim(-2,2)  
                
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

                data_points_1 = ax.scatter(X1_train[0,:], X1_train[1,:], marker='o', c=color_train, label="class 1 train", s=40, cmap=plt.cm.Spectral)
                data_points_2 = ax.scatter(X2_train[0,:], X2_train[1,:], marker='x', c=color_train, label="class 2 train", s=40, cmap=plt.cm.Spectral)

                data_points_3 = ax.scatter(X1_test[0,:], X1_test[1,:], marker='o', c=color_test, label="class 1 test", s=40, cmap=plt.cm.Spectral)
                data_points_4 = ax.scatter(X2_test[0,:], X2_test[1,:], marker='x', c=color_test, label="class 2 test", s=40, cmap=plt.cm.Spectral)

                x = np.arange(-2, 2, 0.01)
                y = -(w_history[j][0]/w_history[j][1])*x - b_history[j]/w_history[j][1]
                line = ax.plot(x, y, color='black', label='Decision Boundary')
                
                legend = ax.legend(loc='upper right')

                return data_points_1,data_points_2,data_points_3, data_points_4,line,legend
            
            frames = len(w_history)
            print(frames)
            animation = FuncAnimation(fig, animate, frames=frames, interval=40, repeat=True)   
            animation.save(f"training_alpha_{alpha}_{k}_3.gif", dpi=300, writer=PillowWriter(fps=25))
            # plt.show()
            
        plt.clf()    
            
        plt.figure()
        plt.plot(i_for_plot, train_err, label="train error")
        plt.plot(i_for_plot, test_err, label="test error")
        plt.legend(loc='upper right')
        # plt.show()
        plt.savefig(f"error_training_alpha_{alpha}_{k}_3.png")
        


if __name__ == "__main__":
    main()
