import numpy as np
import random
from utility import *

def bb_func(x):
    return 13 * x ** 2 + 117 * x

def bb_func_gradient(x):
    return 26 * x + 117

def bb_func_gradient_estimate(x, nz = 100, del_x = 0.001):
    d = 0.0
    px = bb_func(x)
    # Estimate gradient wrt y
    for i in range(nz):
        z = random.gauss(0, 1)
        xz = x + del_x * z
        pxz = bb_func(xz)
        dz = z / (nz * del_x) * (pxz - px)
        d += dz
    return d

#Diabetes ntest = 35
#Aimpeak ntest = 500
def post_process2(exp_name, n_test, n_iter, n_box=10):
    infix = 'colbi_'
    oufix = 'cigar_'
    t = open(result_folder() + oufix + exp_name + "_truth.txt", "r")
    L = t.readlines()
    t.close()
    truth = np.zeros(n_test)
    rmse = np.zeros((n_iter, n_box))
    for i in range(n_test):
        truth[i] = float(L[i])

    for i in range(n_test):
        f = open(result_folder() + infix + exp_name + "_profile" + str(i) + ".txt", "r")
        L = f.readlines()
        f.close()
        for j in range(n_iter):
            tokens = L[j].split(',')
            for k in range(n_box):
                rmse[j, k] += (float(tokens[k]) - truth[i]) ** 2

    for k in range(n_box):
        o = open(result_folder() + oufix + exp_name + "_rmse_box" + str(k) + ".txt", "w+")
        for j in range(n_iter):
            rmse[j, k] = (rmse[j, k] / n_iter) ** 0.5
            o.write(str(rmse[j, k]) + "\n")
        o.close()

    for k in range(n_box):
        o = open(result_folder() + oufix + exp_name + "_nrmse_box" + str(k) + ".txt", "w+")
        for j in range(n_iter):
            o.write(str(rmse[j, k] / np.max(rmse[:,k])) + "\n")
        o.close()

def post_process(exp_name, n_test, n_iter, n_box=10):
    t = open(result_folder() + exp_name + "_truth.txt", "r")
    L = t.readlines()
    t.close()
    truth = np.zeros(n_test)
    rmse = np.zeros((n_iter, n_box))
    for i in range(n_test):
        truth[i] = float(L[i])

    for i in range(n_test):
        f = open(result_folder() + exp_name + "_profile" + str(i) + ".txt", "r")
        L = f.readlines()
        f.close()
        for j in range(n_iter):
            tokens = L[j].split(',')
            for k in range(n_box):
                rmse[j, k] += (float(tokens[k]) - truth[i]) ** 2

    for k in range(n_box):
        o = open(result_folder() + exp_name + "_rmse_box" + str(k) + ".txt", "w+")
        for j in range(n_iter):
            rmse[j, k] = (rmse[j, k] / n_iter) ** 0.5
            o.write(str(rmse[j, k]) + "\n")
        o.close()

    for k in range(n_box):
        o = open(result_folder() + exp_name + "_nrmse_box" + str(k) + ".txt", "w+")
        for j in range(n_iter):
            o.write(str(rmse[j, k] / np.max(rmse[:,k])) + "\n")
        o.close()

# colbi_diabetes_sgpbb
# colbi_diabetes_mixbb
# cigar_diabetes_sgpbb
# cigar_diabetes_mixbb
# cigar_aimpeak_sgpbb
# cigar_aimpeak_mixbb
def main():
    #post_process("colbi_diabetes_sgpbb", 35, 101)
    #post_process("colbi_diabetes_mixbb", 35, 101)
    #post_process("colbi_aimpeak_sgpbb", 500, 101)
    #post_process("colbi_aimpeak_mixbb", 500, 101)
    #post_process2("fullaimpeak_sgpbb", 100, 41)
    #post_process2("fullaimpeak_mixbb", 100, 41)
    #post_process("colbi_protein_sgpbox", 500, 101)
    #post_process("colbi_protein_mixbb", 500, 101)
    #post_process("cigar_diabetes_sgpbb", 35, 41)
    #post_process("cigar_diabetes_mixbb", 35, 41)
    #post_process("cigar_aimpeak_sgpbb", 500, 41)
    #post_process("cigar_aimpeak_mixbb", 500, 41)
    #post_process("colbi_aimpeak_timeslice_sgpbb", 500, 101)
    post_process("colbi_aimpeakfull_timeslice_sgpbb", 500, 101)
    #post_process("cigar_aimpeak_timeslice_sgpbb", 500, 41)

if __name__ == "__main__":
    main()

    # JUNK CODE
    '''
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X, Y, test_size=0.5, random_state=2010)
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test).reshape(X_test.shape[0], 1)
    kmeans = cluster.KMeans(n_clusters=nb, random_state=0).fit(X_train)
    
    Xb = []
    Yb = []
    cs = np.zeros(nb)
    cc = np.zeros(nb)
    for i in range(X_train.shape[0]):
        b = kmeans.labels_[i]
        cs[b] += 1
    for i in range(nb):
        Xb.append(np.zeros((int(cs[i]), X_train.shape[1]), dtype=float))
        Yb.append(np.zeros((int(cs[i]), 1), dtype=float))
    for i in range(X_train.shape[0]):
        b = kmeans.labels_[i]
        Xb[b][int(cc[b])] = X_train[i]
        Yb[b][int(cc[b])] = Y_train[i]
        cc[b] += 1
    '''


    '''
    def dwi(self, x, y, id, w_ext=None):
        if w_ext is None:
            w_ext = self.w

        sigma, bias, coeff = self.extract_params(w_ext)
        d = np.zeros(self.n_params)
        xw = bias + np.dot(x, coeff)[0]
        d[0] = ((y - xw) ** 2) / (sigma ** 2) - 1.0
        d[1] = - 1.0 * (y - xw) / (sigma ** 2)
        for i in range(coeff.shape[0]):
            d[i + 2] = - 1.0 * (y - xw) * x[0, i] / (sigma ** 2)

        return d[id]
    '''


    '''
    def predict_acc(self, x, y):
        sigma, bias, coeff = self.extract_params(self.w)
        xw = bias + np.dot(x, coeff)[0]
        return np.abs(xw - y)
    '''

    '''
    #d[0] = ((y - xw) ** 2) / (sigma ** 2) - 1.0
    #d[0] = - 1.0 * (y - xw) / (sigma ** 2)
    for i in range(self.coeff.shape[0]):
        d[i] = - 1.0 * (y - xw) * x[0, i] / (sigma ** 2)
    '''

    '''
    coeff = np.zeros(ww.shape[0])
    for i in range(coeff.shape[0]):
        coeff[i] = np.exp(ww[i])
    '''

    '''
            for i in range(X1.shape[0]):
                for j in range(X2.shape[0]):
                    if i > j:
                        K[i, j] = K[j, i]
                        continue
                    #for t in range(ls.shape[0]):
                    # U(i, t) = X1(i, t) / ls(t)
                    # V(i, t) = X2(i, t) / ls(t)
                    # A(i) = sum_ t (X1(i, t) ** 2 / (ls(t) ** 2)) ==> A = diag(U * U.t())
                    # B(j) = sum_t (X2(j, t) ** 2 / (ls(t) ** 2)) ==> B = diag(V * V(t))
                    # C(i, j) = sum_t (2 * (X1(i, t) / ls(t)) * (X2(j, t) / ls(t))) ==> C = U * V.t()
                    K[i, j] = np.exp(0.5 * np.sum([- ((X1[i, t] - X2[j, t]) / ls[t]) ** 2 for t in range(ls.shape[0])]))
                    #K[i, j] = np.exp(0.5 * K[i, j])
            K = (signal ** 2) * K
            return K
            '''