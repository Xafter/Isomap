from numpy import *
import numpy
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.io as sio
from queue import PriorityQueue as pq


def get_Graph(data, k):
    N = data.shape[0]
    G = numpy.zeros((N, N))
    BOOL = G == 0
    G[BOOL] = inf
    for i in range(N):
        LEN = data.shape[0]
        dis = data - numpy.tile(data[i], (LEN, 1))
        dis = numpy.linalg.norm(dis, axis=1)
        order = numpy.argsort(dis)
        dis = numpy.sort(dis)
        dist, order = dis[1:k + 1], order[1:k + 1]
        G[i, order] = dist
        G[order, i] = dist
        G[i, i] = 0
    return G


def c_path(G, v):
    D = numpy.copy(G[v])
    P = [-1 for i in range(D.shape[0])]
    final = [0 for i in range(D.shape[0])]
    final[v] = 1
    Qu = pq()
    for i in range(D.shape[0]):
        Qu.put((D[i], i))
    D[v]=0
    for i in range(D.shape[0]):
        if sum(final) == D.shape[0]:
            break
        if i != v:
            ite = Qu.get()
            key = ite[1]
            while final[key] != 0:
                ite = Qu.get()
                key = ite[1]
            k = key
            mini = D[key]
            final[k] = 1
            for j in range(D.shape[0]):
                if final[j] == 0 and (mini + G[k, j] < D[j]):
                    D[j] = mini + G[k, j]
                    Qu.put((D[j], j))
                    P[j] = k
    pattern = {-1:v}
    P = [x if x not in pattern else pattern[x] for x in P]
    return D, P, final


def Mds_projection(G, q):
    G = asarray(G)
    D = G.copy()
    for i in range(G.shape[0]):
        D_, P, final = c_path(G, i)
        D[i, :] = D_
    DSquare = D ** 2
    totalMean = mean(DSquare)
    columnMean = mean(DSquare, axis=0)
    rowMean = mean(DSquare, axis=1)
    B = zeros(DSquare.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = -0.5 * (DSquare[i][j] - rowMean[i] - columnMean[j] + totalMean)
    eigVal, eigVec = linalg.eig(B)
    X = dot(eigVec[:, :q], sqrt(diag(eigVal[:q])))
    return X

def Isomap(data, K, d):
    G = get_Graph(data, K)
    N = data.shape[0]
    G_ = G.copy()
    for i in range(N):
        D, P, final = c_path(G, i)
        G_[i] = D
    Y = Mds_projection(G_, d)
    return Y


if __name__ == '__main__':
    '''
    save_path = "./synthetic_result"
    path = "./synthetic_data"
    data_name = "Swiss_Roll"
    suffix = "_1000.mat"
    data_path = path + r"/" + data_name + suffix
    MAT = sio.loadmat(data_path)
    data = MAT["X"]
    color = MAT["ColorVector"]
    K = 12
    d = 2
    Y = Isomap(data, K, d)
    fig = plt.figure(1)
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.jet)
    plt.show()
    '''
    path = './mnist_data'
    data_name = 'usps_4000.mat'
    save_path = './mnist_result'
    data_path = path + './' + data_name
    MAT = sio.loadmat(data_path)
    trset = MAT['trset'][:, 0:-1]
    teset = MAT['teset'][:, 0:-1]
    data = vstack((trset, teset))
    ds = [20, 50, 80, 100]
    K = 10
    for d in ds:
        Y = Isomap(data, K, d)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        sio.savemat(save_path + r"/" + str(d) + 'D' + data_name, {'Y': Y})
    print("complete")