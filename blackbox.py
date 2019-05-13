import matplotlib
matplotlib.use('PS')
import numpy as np
from GPy import kern, models
from sklearn import linear_model
import random
import keras

class BlackBox:
    def __init__(self, X, Y):
        #do something
        self.X = X
        self.Y = Y
        random.seed(2010)

class BRR_BlackBox(BlackBox):
    def __init__(self, X, Y):
        BlackBox.__init__(self, X, Y)
        self.clf = linear_model.BayesianRidge(n_iter = 200)
        self.clf.fit(X, Y)

    def query(self, x, y):
        ym, yv = self.clf.predict(x, return_std=True)
        pr = -0.5 * ((y - ym) ** 2) / (yv ** 2)
        return np.log(1.0 / (np.sqrt(2.0 * np.pi) * yv)) + pr

    def predict(self, x):
        ym = self.clf.predict(x)
        return ym

    def dy(self, x, y, nz=10, del_y = 0.001):
        d = 0.0
        py = self.query(x, y)
        # Estimate gradient wrt y
        for i in range(nz):
            z = random.gauss(0, 1)
            yz = y + del_y * z
            pyz = self.query(x, yz)
            dz = z / (nz * del_y) * (pyz - py)
            d += dz
        return d

class SGP_BlackBox(BlackBox):
    def __init__(self, X, Y, Z):
        BlackBox.__init__(self, X, Y)
        #Z = 2.0 * np.random.rand(int(np.sqrt(X.shape[0])), X.shape[1])
        K = kern.RBF(X.shape[1], 1.0, 1.0 * np.ones(X.shape[1]), ARD=True)
        self.m = models.SparseGPRegression(X, Y, Z=Z, kernel=K)
        self.m.optimize('bfgs', max_iters = 200)

    def query(self, x, y):
        pred = self.m.predict(x)
        mean = pred[0][0][0]
        sigma = pred[1][0][0]
        pr = -0.5 * ((y - mean) ** 2) / (sigma ** 2)
        return np.log(1.0 / (np.sqrt(2.0 * np.pi) * sigma)) + pr

    def predict(self, x):
        pred = self.m.predict(x)
        return pred[0][0][0]

    def predict_acc(self, x, y):
        pred = self.m.predict(x)
        return np.abs(y - pred[0][0][0])

    def dy(self, x, y, nz=10, del_y=0.001):
        d = 0.0
        py = self.query(x, y)
        # Estimate gradient wrt y
        for i in range(nz):
            z = random.gauss(0, 1)
            yz = y + del_y * z
            pyz = self.query(x, yz)
            dz = z / (nz * del_y) * (pyz - py)
            d += dz
        return d

    def true_dy(self, x, y):
        pred = self.m.predict(x)
        mean = pred[0][0][0]
        sigma = pred[1][0][0]
        return (mean - y) / (sigma ** 2)