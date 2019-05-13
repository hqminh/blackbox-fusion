from sklearn import datasets, model_selection, cluster
from blackbox import *
from utility import *
import random
import time
import copy

def fuse_prediction(exp_name, box, X, YY, n_iter = 40, lr = 1):
    pred = np.zeros((X.shape[0], len(box)))
    g  = open(result_folder() + exp_name + "_test.txt", "w+")
    h  = open(result_folder() + exp_name + "_truth.txt", "w+")
    tm = open(result_folder() + exp_name + "_dectime.txt", "w+")
    dec_time = 0
    lin_time = 0
    for i in range(X.shape[0]):
        # init y-candidates
        t1 = time.time()
        f = open(result_folder() + exp_name + "_profile" + str(i) +".txt", "w+")
        x = X[i].reshape((1, X.shape[1]))
        for j in range(X.shape[1]):
            g.write(str(x[0, j]))
            if j == X.shape[1] - 1:
                g.write("\n")
            else:
                g.write(",")
        h.write(str(YY[i, 0]) + "\n")
        ybox = np.zeros(len(box))
        print("Fusing prediction for test point #" + str(i))
        t2 = time.time()
        for j in range(len(box)):
            ybox[j] = random.gauss(box[j].predict(x), 2.0)
            f.write(str(ybox[j]))
            if j == len(box) - 1:
                f.write("\n")
            else:
                f.write(",")
        print("Initial guess " + str(ybox))
        for t in range(n_iter):
            dybox = np.zeros(len(box))
            t41 = time.time()
            for j1 in range(len(box)):
                for j2 in range(len(box)):
                    dybox[j1] += box[j2].dy(x, ybox[j1])
            for j in range(len(box)):
                ybox[j] += lr * dybox[j] / (np.linalg.norm(dybox[j]) * np.sqrt(t + 1))
                f.write(str(ybox[j]))
                if j == len(box) - 1:
                    f.write("\n")
                else:
                    f.write(",")
        print("Post-fusion " + str(ybox))
        t4 = time.time()
        for j in range(len(box)):
            pred[i, j] = ybox[j]
        f.close()
        dec_time += (t4 - t2) / len(box) + (t2 - t1)
        lin_time += t4 - t1
    g.close()
    tm.write(str(dec_time) + " " + str(lin_time))
    tm.close()
    return pred

def experiment_8():
    X, Y = load_traffic()
    nb = 10
    box_type = []
    for i in range(nb):
        if i < 5:
            box_type.append("SGP")
        else:
            box_type.append("BRR")
    cigar(X, Y, "cigar_fullaimpeak_mixbb", box_type)


def experiment_7():
    X, Y = load_traffic()
    nb = 10
    box_type = []
    for i in range(nb):
        box_type.append("SGP")
    cigar(X, Y, "cigar_fullaimpeak_sgpbb", box_type)

def experiment_6():
    X, Y = load_protein()
    nb = 10
    box_type = []
    for i in range(nb):
        if i < 5:
            box_type.append("SGP")
        else:
            box_type.append("BRR")
    cigar(X, Y, "cigar_protein_mixbb", box_type)

def experiment_5():
    X, Y = load_protein()
    nb = 10
    box_type = []
    for i in range(nb):
        box_type.append("SGP")
    cigar(X, Y, "cigar_protein_sgpbb", box_type)

# AIMPEAK mixed black-boxes
def experiment_4():
    X, Y = load_traffic()
    nb = 10
    box_type = []
    for i in range(nb):
        box_type.append("SGP")

    cigar(X, Y, "cigar_aimpeak_sgpbb", box_type)

# AIMPEAK mixed black-boxes
def experiment_3():
    X, Y = load_traffic()
    nb = 10
    box_type = []
    for i in range(nb):
        if i < 5:
            box_type.append("BRR")
        else:
            box_type.append("SGP")

    cigar(X, Y, "cigar_aimpeak_mixbb", box_type)

# Diabetes mixed black-boxes
def experiment_2():
    X, Y = datasets.load_diabetes(return_X_y=True)
    nb = 10
    box_type = []
    for i in range(nb):
        if i < 5:
            box_type.append("SGP")
        else:
            box_type.append("BRR")
    cigar(X, Y, "cigar_diabetes_mixbb", box_type)


# Diabetes only sgp black-boxes
def experiment_1():
    X, Y = datasets.load_diabetes(return_X_y=True)
    nb = 10
    box_type = []
    for i in range(nb):
        box_type.append("SGP")
    cigar(X, Y, "cigar_diabetes_sgpbb", box_type)

def experiment_9():
    X, Y = load_traffic()
    nb = 10
    box_type = []
    for i in range(nb):
        box_type.append("SGP")

    cigar2(X, Y, "cigar_aimpeak_timeslice_sgpbb", box_type, box_size=50)

def cigar2(X, Y, exp_name, box_type, box_size=500, test_size=500):
    nb = len(box_type)
    kf = model_selection.KFold(n_splits= nb + 2)
    Xb = []
    Yb = []
    slice_size = 775
    slice_per_box = 5
    offset = 0
    for i in range(nb):
        train_index = list(range(offset, offset + slice_size * slice_per_box))
        Xb.append(X[train_index])
        Yb.append(Y[train_index].reshape((slice_size * slice_per_box, 1)))
        offset += slice_size * slice_per_box
    test_index = list(range(offset, offset + slice_size * 2))
    Xt = copy.deepcopy(X[test_index])
    Yt = copy.deepcopy(Y[test_index].reshape((slice_size * 2, 1)))
    offset += slice_size * 2
    Xs = copy.deepcopy(X[offset:])

    '''
    fold_no = 0
    for train_index, test_index in kf.split(X):
        if fold_no < nb:
            Xb.append(X[test_index])
            Yb.append(Y[test_index].reshape((len(test_index), 1)))

        elif fold_no == nb:
            Xs = X[test_index]
            Ys = Y[test_index].reshape((len(test_index), 1))
        else:
            Xt = X[test_index]
            Yt = Y[test_index].reshape((len(test_index), 1))
        fold_no += 1
    '''

    id = np.random.choice(range(Xs.shape[0]), min(box_size, Xs.shape[0]), replace=False)
    Xs = Xs[id, :]

    id = np.random.choice(range(Xt.shape[0]), min(test_size, Xt.shape[0]), replace=False)
    Xt = Xt[id, :]
    Yt = Yt[id, :]

    '''
    f = open(result_folder() + exp_name + "_truth.txt", "w+")
    for i in range(Yt.shape[0]):
        f.write(str(Yt[i, 0]) + "\n")
    return
    '''

    box = []
    for b in range(nb):
        print("Setting up Black-Box #" + str(b))
        if box_type[b] == 'SGP':
            id = np.random.choice(range(Xb[b].shape[0]), min(box_size, Xb[b].shape[0]), replace=False)
            Xb[b] = Xb[b][id, :]
            Yb[b] = Yb[b][id, :]
            bx = SGP_BlackBox(Xb[b], Yb[b], Xs)
        elif box_type[b] == 'BRR':
            id = np.random.choice(range(Xb[b].shape[0]), min(box_size, Xb[b].shape[0]), replace=False)
            Xb[b] = Xb[b][id, :]
            Yb[b] = Yb[b][id, :]
            bx = BRR_BlackBox(Xb[b], Yb[b].reshape(Yb[b].shape[0]))
        box.append(bx)

    f = open(result_folder() + exp_name + "_fusion_rmse.txt", "w+")
    pre = np.zeros(nb)
    for b in range(nb):
        print("Testing Black-Box #" + str(b))
        #box_rmse = 0.0
        for j in range(Xt.shape[0]):
            x = Xt[j].reshape(1, Xt.shape[1])
            box_pred = box[b].predict(x)
            pre[b] += (box_pred - Yt[j, 0]) ** 2
        pre[b] = (pre[b] / Xt.shape[0]) ** 0.5
        print("Box RMSE = " + str(pre[b]))
    pos = np.zeros(nb)
    pred = fuse_prediction(exp_name, box, Xt, Yt)
    for b in range(nb):
        print("Post-fusion prediction #" + str(b))
        #box_rmse = 0.0
        for j in range(Xt.shape[0]):
            pos[b] = (pred[j, b] - Yt[j, 0]) ** 2
        pos[b] = (pos[b] / Xt.shape[0]) ** 0.5
        print("Box RMSE = " + str(pos[b]))
        f.write(str(pre[b]) + "," + str(pos[b]) + "\n")

    Xtrain = Xb[0]
    Ytrain = Yb[0]
    for b in range(nb - 1):
        Xtrain = np.vstack((Xtrain, Xb[b + 1]))
        Ytrain = np.vstack((Ytrain, Yb[b + 1]))
    master_box = SGP_BlackBox(Xtrain, Ytrain, Xs)
    print("Testing Master Black-Box")
    box_rmse = 0.0
    for j in range(Xt.shape[0]):
        x = Xt[j].reshape(1, Xt.shape[1])
        box_pred = master_box.predict(x)
        box_rmse += (box_pred - Yt[j, 0]) ** 2
    box_rmse = (box_rmse / Xt.shape[0]) ** 0.5
    print("Box RMSE = " + str(box_rmse))
    f.write(str(box_rmse) + "\n")
    f.close()


def cigar(X, Y, exp_name, box_type, box_size=500, test_size=500):
    nb = len(box_type)
    kf = model_selection.KFold(n_splits= nb + 2)
    Xb = []
    Yb = []
    fold_no = 0
    for train_index, test_index in kf.split(X):
        if fold_no < nb:
            Xb.append(X[test_index])
            Yb.append(Y[test_index].reshape((len(test_index), 1)))

        elif fold_no == nb:
            Xs = X[test_index]
            Ys = Y[test_index].reshape((len(test_index), 1))
        else:
            Xt = X[test_index]
            Yt = Y[test_index].reshape((len(test_index), 1))
        fold_no += 1

    id = np.random.choice(range(Xs.shape[0]), min(box_size, Xs.shape[0]), replace=False)
    Xs = Xs[id, :]

    id = np.random.choice(range(Xt.shape[0]), min(test_size, Xt.shape[0]), replace=False)
    Xt = Xt[id, :]
    Yt = Yt[id, :]

    '''
    f = open(result_folder() + exp_name + "_truth.txt", "w+")
    for i in range(Yt.shape[0]):
        f.write(str(Yt[i, 0]) + "\n")
    return
    '''

    box = []
    for b in range(nb):
        print("Setting up Black-Box #" + str(b))
        if box_type[b] == 'SGP':
            id = np.random.choice(range(Xb[b].shape[0]), min(box_size, Xb[b].shape[0]), replace=False)
            Xb[b] = Xb[b][id, :]
            Yb[b] = Yb[b][id, :]
            bx = SGP_BlackBox(Xb[b], Yb[b], Xs)
        elif box_type[b] == 'BRR':
            id = np.random.choice(range(Xb[b].shape[0]), min(box_size, Xb[b].shape[0]), replace=False)
            Xb[b] = Xb[b][id, :]
            Yb[b] = Yb[b][id, :]
            bx = BRR_BlackBox(Xb[b], Yb[b].reshape(Yb[b].shape[0]))
        box.append(bx)

    f = open(result_folder() + exp_name + "_fusion_rmse.txt", "w+")
    pre = np.zeros(nb)
    for b in range(nb):
        print("Testing Black-Box #" + str(b))
        #box_rmse = 0.0
        for j in range(Xt.shape[0]):
            x = Xt[j].reshape(1, Xt.shape[1])
            box_pred = box[b].predict(x)
            pre[b] += (box_pred - Yt[j, 0]) ** 2
        pre[b] = (pre[b] / Xt.shape[0]) ** 0.5
        print("Box RMSE = " + str(pre[b]))
    pos = np.zeros(nb)
    pred = fuse_prediction(exp_name, box, Xt, Yt)
    for b in range(nb):
        print("Post-fusion prediction #" + str(b))
        #box_rmse = 0.0
        for j in range(Xt.shape[0]):
            pos[b] = (pred[j, b] - Yt[j, 0]) ** 2
        pos[b] = (pos[b] / Xt.shape[0]) ** 0.5
        print("Box RMSE = " + str(pos[b]))
        f.write(str(pre[b]) + "," + str(pos[b]) + "\n")

    Xtrain = Xb[0]
    Ytrain = Yb[0]
    for b in range(nb - 1):
        Xtrain = np.vstack((Xtrain, Xb[b + 1]))
        Ytrain = np.vstack((Ytrain, Yb[b + 1]))
    master_box = SGP_BlackBox(Xtrain, Ytrain, Xs)
    print("Testing Master Black-Box")
    box_rmse = 0.0
    for j in range(Xt.shape[0]):
        x = Xt[j].reshape(1, Xt.shape[1])
        box_pred = master_box.predict(x)
        box_rmse += (box_pred - Yt[j, 0]) ** 2
    box_rmse = (box_rmse / Xt.shape[0]) ** 0.5
    print("Box RMSE = " + str(box_rmse))
    f.write(str(box_rmse) + "\n")
    f.close()


def main():
    random.seed(2010)
    #experiment_1()
    #experiment_2()
    #experiment_3()
    #experiment_4()
    #experiment_5()
    #experiment_6()
    #experiment_7()
    #experiment_8()
    experiment_9()

if __name__ == "__main__":
    main()