import numpy
from numpy import *
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


#计算某个点的k个近邻点
#IN
#data:数据
#k:近邻点数
#x:待计算点的index
#OUT
#dist:x与其k个近邻点的距离
#order:x的k个近邻点的index
def calc_G(data, k,x):
    LEN = data.shape[0]
    dis=data-numpy.tile(data[x],(LEN,1))
    dis = numpy.linalg.norm(dis,axis=1)
    order=numpy.argsort(dis)
    dis=numpy.sort(dis)
    return dis[1:k+1],order[1:k+1]

#获取visible neighbors
#IN
#data:数据集
#K：近邻点数
#x:待计算点的index
#OUT
#dist：x与其visible neighbours之间的距离
#order：x的visible neighbours的index


def get_visible(data,K,x):
    dist,order=calc_G(data, K, x)
    dist=list(dist)
    order=list(order)
    order_copy = order.copy()
    for i in range(data[order].shape[0]):
        y = data[order[i]]
        for z in data[order]:
            if numpy.dot(data[x]-z, y-z)<0:
                order_copy[i]=-1
                break
    for i in range(len(order)):
        if order_copy[i]==-1:
            order[i]=-1
            dist[i]=-1
    while -1 in order:
        order.remove(-1)
    while -1 in dist:
        dist.remove(-1)
    return dist,order
#本征维度估计
#IN
#data:edges，待计算点与其visible neighbour之间的edges
#OUT
#d:本征维度


def demension_estimate(data):
    mean=numpy.mean(data,axis=0)
    data_mean=data-mean
    covmat=numpy.cov(data_mean,rowvar=0)
    eigenvalue,eigenvector=numpy.linalg.eig(numpy.mat(covmat))
    p = eigenvalue/max(eigenvalue)
    d=numpy.sum(p > 0.08)
    return d

#获取安全neighbour
#IN
#data:edges
#dist:各点与待计算点之间的距离
#order:各点的index
#p:超参数

'''
def get_safeNeighb(data,dist,order,p):
    dist_copy = dist.copy()
    data_shape=data.shape[0]
    Dimensions=[]
    for i in range(1,data_shape):
        d=demension_estimate(data[0:i+1])
        Dimensions.append(d)
    length = len(Dimensions)
    for j in range(1,length):
        if Dimensions[j]>Dimensions[j-1]:
            delta = numpy.linalg.norm(data[j])-numpy.linalg.norm(data[j-1])
            if delta > p*numpy.mean(dist_copy[0:j]):
                dist[j:] = [-1]*len(dist[j:])
                order[j:] = [-1]*len(order[j:])
                break
    while -1 in dist:
        dist.remove(-1)
    while -1 in order:
        order.remove(-1)
    return dist, order
'''
def get_safeNeighb(data, dist, order, x):
    dist1, order1 = calc_G(data, 10, x)
    for i in range(len(order)):
        if order[i] not in order1:
            order[i] = -1
            dist [i] = -1
    while -1 in order:
        order.remove(-1)
        dist.remove(-1)
    return dist, order
#计算邻接矩阵
#IN
#data：数据


def get_Graph(data, K):
    N=data.shape[0]
    G=numpy.zeros((N,N))
    BOOL=G==0
    G[BOOL]=inf
    for i in range(N):
        print("Get visible neighbour of point:%d" %(i))
        dist, order=get_visible(data, K, i)
        #print("Get ""safe"" neighbour of point:%d" %(i))
        #dist, order = get_safeNeighb(data, dist, order,i)
        #dist, order = calc_G(data, K, i)   #knn
        G[i,order] = dist
        G[order,i] = dist
        G[i, i] = 0
    return G


if __name__=='__main__':
    G=get_Graph(x)
    N = x.shape[0]
    fig = plt.figure(1)
    ax = fig.gca(projection="3d")
    #for i in range(N):
    #    region = nonzero(G[i] != inf)[0]
    #    for each in region:
    #        ax.plot([x[i, 0], x[each, 0]], [x[i, 2], x[each, 2]], [x[i, 1], x[each, 1]])
    ax.scatter(x[:, 0], x[:, 2],x[:, 1])
    plt.show()

