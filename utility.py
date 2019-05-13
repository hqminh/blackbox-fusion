import numpy as np

def load_traffic():
    f = open("aimpeak.csv","r")
    l = f.readlines()
    X = np.zeros((len(l), 5))
    Y = np.zeros((len(l), 1))
    for i in range(len(l)):
        tokens = l[i].split(',')
        for j in range(X.shape[1]):
            X[i, j] = float(tokens[j + 1])
        Y[i, 0] = float(tokens[6])

    '''    
    for j in range(X.shape[1]):
        X[:, j] = X[:, j] / np.linalg.norm(X[:, j])
    '''
    return X, Y

def load_protein():
    f = open("protein.csv", "r")
    l = f.readlines()
    X = np.zeros((len(l), 9))
    Y = np.zeros((len(l), 1))
    for i in range(len(l)):
        tokens = l[i].split(',')
        for j in range(X.shape[1]):
            X[i, j] = float(tokens[j])
        Y[i, 0] = float(tokens[9])
    return X, Y

def result_folder():
    return "./result/"
