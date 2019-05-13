import numpy as np
import time
from sklearn import datasets
import random, math

class Surrogate:
    def __init__(self, n_params):
        self.w = np.zeros(n_params)
        self.n_params = n_params

    def update(self, i, dw, lr=1):
        self.w = self.w - lr / np.sqrt(i + 1) * dw

class GPSurrogate(Surrogate):
    def __init__(self, n_params, X, Y):
        Surrogate.__init__(self, n_params)
        self.X = X
        self.Y_mean = np.mean(Y)
        self.Y = Y - self.Y_mean
        self.KXX = np.eye(X.shape[0])
        self.KXX_inv = np.eye(X.shape[0])

    def extract_params(self, ww):
        noise = np.exp(ww[0])
        signal = np.exp(ww[1])
        ls = np.zeros(ww.shape[0] - 2)
        for i in range(ls.shape[0]):
            ls[i] = np.exp(ww[i + 2])
        return noise, signal, ls

    def update(self, i, dw, lr=1):
        Surrogate.update(self, i, dw, lr)
        self.precompute_inv()

    def precompute_inv(self):
        noise, signal, ls = self.extract_params(self.w)
        self.KXX = self.kernel2(self.X, self.X, signal, ls)
        self.KXX_inv = np.linalg.inv(self.KXX + (noise ** 2) * np.eye(self.X.shape[0]))

    def kernel(self, X1, X2, signal, ls):
        A = np.zeros((X1.shape[0], X1.shape[1]))
        B = np.zeros((X2.shape[0], X2.shape[1]))
        A[:] = X1[:] * ls
        B[:] = X2[:] * ls
        B = np.transpose(B)
        oA = np.ones((X1.shape[0], X1.shape[1]))
        oB = np.ones((X2.shape[1], X2.shape[0]))
        K = np.exp(-0.5 * (signal ** 2) * np.dot((A ** 2), oB),   - 0.5 * (signal ** 2) * np.dot(oA,(B ** 2)) + (signal ** 2) * np.dot(A, B))
        return K

    def kernel2(self, X1, X2, signal, ls):
        K = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                if i > j:
                    K[i, j] = K[j, i]
                    continue
                # for t in range(ls.shape[0]):
                # U(i, t) = X1(i, t) / ls(t)
                # V(i, t) = X2(i, t) / ls(t)
                # A(i) = sum_ t (X1(i, t) ** 2 / (ls(t) ** 2)) ==> A = diag(U * U.t())
                # B(j) = sum_t (X2(j, t) ** 2 / (ls(t) ** 2)) ==> B = diag(V * V(t))
                # C(i, j) = sum_t (2 * (X1(i, t) / ls(t)) * (X2(j, t) / ls(t))) ==> C = U * V.t()
                K[i, j] = np.exp(0.5 * np.sum([- ((X1[i, t] - X2[j, t]) / ls[t]) ** 2 for t in range(ls.shape[0])]))
                # K[i, j] = np.exp(0.5 * K[i, j])
        K = (signal ** 2) * K

        return K

    def proba(self, x, y, w_ext=None):
        if w_ext is None:
            w_ext = self.w
        ym, yv = self.predict(x, compute_var=True, w_ext=w_ext)
        pr = -0.5 * ((y - ym) ** 2) / yv
        return np.log(1.0 / (np.sqrt(2.0 * np.pi) * (yv ** 0.5))) + pr

    def predict(self, x, compute_var=False, compute_d=False, w_ext=None):
        if w_ext is None:
            w_ext = self.w
        noise, signal, ls = self.extract_params(self.w)
        ktX = self.kernel2(x, self.X, signal, ls)
        temp1 = np.dot(ktX, self.KXX_inv)
        temp2 = np.dot(self.KXX_inv, self.Y)
        ypred = np.dot(temp1, self.Y) + self.Y_mean

        if compute_var:
            ktt = self.kernel2(x, x, signal, ls)
            yvar = ktt - np.dot(temp1, np.transpose(ktX))
            if yvar[0, 0] < 0:
                print(str(ktX))
                print(str(ktt))
                input("Enter!")
            if not compute_d:
                return ypred[0, 0], yvar[0, 0]

        if compute_d:
            dm = np.zeros(self.n_params)
            dn = np.dot(temp1, temp2)
            dm[0] = - 2.0 * noise * dn[0, 0]
            dm[1] = - (noise / signal) * dm[0]

            if compute_var:
                dv = np.zeros(self.n_params)
                dnv = np.dot(temp1, np.transpose(temp1))
                dv[0] = 2 * noise * dnv[0, 0]
                dv[1] = (2.0 / signal) * yvar - (noise / signal) * dv[0]

            for i in range(self.n_params - 2):
                dktX = np.zeros((1, self.X.shape[0]))
                dkXX = np.zeros((self.X.shape[0], self.X.shape[0]))
                for j in range(self.X.shape[0]):
                    dktX[0, j] = ktX[0, j] * ((x[0, i] - self.X[j, i]) ** 2) / (ls[i] ** 3)
                    for k in range(self.X.shape[0]):
                        dkXX[j, k] = self.KXX[j, k] * ((self.X[j, i] - self.X[k, i]) ** 2) / (ls[i] ** 3)
                dm[i + 2] = (np.dot(dktX, temp2) - np.dot(temp1, np.dot(dkXX, temp2)))[0, 0]
                if compute_var:
                    dv[i + 2] = (np.dot(dktX, np.transpose(temp1)) - np.dot(temp1, np.dot(dkXX, np.transpose(temp1))))[0, 0]

            if compute_var:
                return ypred[0, 0], yvar[0, 0], dm, dv

            else:
                return ypred[0, 0], dm

        return ypred[0, 0]

    def predict_acc(self, x, y):
        yw = self.predict(x)
        return np.abs(yw - y)

    def dw(self, x ,y, w_ext=None):
        if w_ext is None:
            w_ext = self.w

        ym, yv, dm, dv = self.predict(x, compute_var=True, compute_d=True, w_ext=w_ext)
        dp_dm = (y - ym) / yv
        dp_dv = 0.5 * (y - ym) ** 2 / (yv ** 2) - 0.5 / yv

        d = np.zeros(self.n_params)
        for i in range(self.n_params):
            d[i] = (dp_dm * dm[i] + dp_dv * dv[i]) * np.exp(w_ext[i])

        return d

    def dy(self, x, y):
        ym, yv = self.predict(x, compute_var=True, compute_d=False)
        return (ym - y) / yv

    def dy_est(self, x, y, nz=10, del_y=0.001):
        d = 0.0
        py = self.proba(x, y)
        # Estimate gradient wrt y
        for i in range(nz):
            z = random.gauss(0, 1)
            yz = y + del_y * z
            pyz = self.proba(x, yz)
            dz = z / (nz * del_y) * (pyz - py)
            d += dz
            if math.isnan(d):
                ym, yv = self.predict(x, compute_var=True)
                print(str(y) + " " + str(yz) + " " + str(ym) + " " + str(yv))
                print(str(z) + " " + str(py) + " " + str(pyz))
                print(str(self.w))
                input("Enter")
        return d

class LinearSurrogate(Surrogate):
    def __init__(self, n_params):
        Surrogate.__init__(self, n_params)

    def extract_params(self, ww):
        #sigma = np.exp(self.sigma)
        #bias = np.exp(self.bias)
        coeff = np.asarray(map(np.exp, ww))
        return coeff

    def update(self, i, dw, lr=1):
        Surrogate.update(self, i, dw, lr)
        self.unfold()

    def unfold(self):
        self.coeff = np.fromiter(map(np.exp, self.w), dtype=float)

    def proba(self, x, y):
        #sigma, bias, coeff = self.extract_params(self.w)
        xw = self.bias + np.dot(x, self.coeff)[0]
        pr = np.exp(-0.5 * ((y - xw) ** 2) / (self.sigma ** 2))
        return 1.0 / (np.sqrt(2.0 * np.pi) * self.sigma) * pr

    def predict(self, x):
        #sigma, bias, coeff = self.extract_params(self.w)
        return self.bias + np.dot(x, self.coeff)[0]

    def dw(self, x, y, w_ext=None):
        if w_ext is None:
            w_ext = self.w

        #sigma, bias, coeff = self.extract_params(w_ext)
        d = np.zeros(self.n_params)
        xw = self.bias + np.dot(x, self.coeff)[0]
        d[:] = - 1.0 * (y - xw) * x[0, :] / (self.sigma ** 2)

        return d

    def dy(self, x, y):
        #sigma, bias, coeff = self.extract_params(self.w)
        xw = self.bias + np.dot(x, self.coeff)[0]
        return (xw - y) / (self.sigma ** 2)
