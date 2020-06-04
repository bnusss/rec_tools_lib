#!/usr/bin/env python
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import time
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)

def generate_network():
    dg = nx.DiGraph()
    # add nodes
    for i in range(100):
        dg.add_node(i)
    path = './insilico_size100_3_goldstandard.tsv'
    # 读取文件

    df = pd.read_csv(path, header=None, sep='\t')

    for i in range(len(df)):
        if (int(df.at[i, 2]) != 0):
            first = int(df.at[i, 0].lstrip('G')) - 1
            second = int(df.at[i, 1].lstrip('G')) - 1
            dg.add_edge(first, second)
    print('生成的是基因网')
    print(len(dg.edges()))
    return dg


# 计算每个节点的入度
def cal_degree(matrix):
    return np.sum(matrix, axis=0)


# 找到每个节点和他相连的邻居
# return:{0:[1,2,3],1:[0,4]...}
def get_innode(adj):
    innodes = {}
    for i in range(adj.shape[0]):
        innode = []
        for j in range(adj.shape[0]):
            if adj[j][i] == 1:
                innode.append(j)
        innodes[i] = innode
    return innodes


graph = generate_network()
# 生成对应的邻接矩阵
edges = nx.adjacency_matrix(graph).toarray()
np.savetxt('/data/zhangyan/Real net&dyn/gene/menten_connectivity_1000.dat', edges, fmt='%.4f', delimiter='\t')
# 每个节点的入度
Ni = cal_degree(edges)
print(Ni)
# 每个节点的邻居
neibour = get_innode(edges)


def michaelis_menten(y, t):
    dydt = np.zeros((y.shape[0],))
    # J = np.loadtxt('Data/connectivity.dat')

    k = edges.shape[0]

    for i in range(k):
        if Ni[i] == 0:
            # print('节点度为0'+str(i))
            dydt[i] = -y[i]
        else:
            sum = 0.0
            for j in neibour[i]:
                sum += (edges[j][i] * (y[j] / (1 + y[j])))
            dydt[i] = -y[i] + sum * (1 / Ni[i])

    return (dydt)


def simulate(MODEL, N, S, M):
    '''
     simulate(MODEL,N,NI,S,M) generates time series of networks of dynamical
     systems for several different intial conditions.

     Parameters
     ------------------
     MODEL: Dynamical model on network units. Currently, only kuramoto1,
            kuramoto2, michaelis_menten and roessler are supported. For
            detailed information about the models, please check methods
            section in the main manuscript.
     N:     Network size.
     NI:    Number of incoming connections per unit.
     S:     Number of different time series.
     M:     Number of time points per time series.

     Input type
     ------------------
     MODEL: string
     N:     integer
     NI:    integer (NI<N)
     S:     integer
     M:     integer

     Output
     ------------------
     'Data/data.dat':     File containing all simulated time series in a
                          concatenaded form.
     'Data/ts_param.dat': File containing time series parameters, i.e. S and
                          M, for later extracting the different time series.

     Example
     ------------------
     simulate('kuramoto2',25,4,30,10) generates 30 time series of 10 time
     points each for a network of 25 oscillators defined by the model
     kuramoto2. Each oscillator has 4 incoming connections.

     Accompanying material to "Model-free inference of direct interactions
     from nonlinear collective dynamics".

     Author: Jose Casadiego
     Date:   May 2017
    '''

    # smpling rate of time series
    print('Simulating time series...')
    Y = np.array([])
    resolution = 1

    if MODEL == 'michaelis_menten':
        for s in range(S):
            init = 1 + np.random.uniform(0., 1., size=(N,))
            tspan = np.arange(0, M, resolution)
            y = odeint(michaelis_menten, init, tspan)
            # print(y)
            Y = np.vstack((Y, y)) if Y.size else y

    ts_param = [S, M]
    np.savetxt('/data/zhangyan/Real net&dyn/gene/menten_data_1000.dat', Y, fmt='%.4f', delimiter='\t')
    np.savetxt('/data/zhangyan/Real net&dyn/gene/menten_ts_param_1000.dat', ts_param, fmt='%i', delimiter='\t')

    print('Simulation finished!')
    print(Y.shape)

if __name__ == '__main__':
    simulate('michaelis_menten', 100, 1000, 10)


end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('end_time:', end_time)
