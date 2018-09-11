from part1 import *
import numpy as np

def load_dataset(filename='credit.csv'):
    '''
        Load dataset 1 from the CSV file: 'data1.csv'.
        The first row of the file is the header (including the names of the attributes)
        In the remaining rows, each row represents one data instance.
        The first column of the file is the label to be predicted.
        In remaining columns, each column represents an attribute.
        Input:
            filename: the filename of the dataset, a string.
        Output:
            X: the feature matrix, a numpy matrix of shape p by n.
               Each element can be int/float/string.
               Here n is the number data instances in the dataset, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    alldata = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=str)

    X = alldata[:, 1:-1].T
    Y = alldata[:, -1].T

    #########################################
    return X, Y

X, Y = load_dataset()

t = Tree.train(X, Y)

