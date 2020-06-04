#!/usr/bin/env python
from math import pi
import numpy as np
from sklearn.metrics import roc_curve, auc
import sys
import matplotlib.pyplot as plt
from basis_expansion import basis_expansion
import time

start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)

def calc_tptnfpfn(out,adj):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(out.shape[0]):
        for j in range(out.shape[0]):
            if adj[i][j] == 1:
                # positive
                if out[i][j] == 1:
                    # true positive
                    tp += 1
                else:
                    # false positive
                    fn += 1
            else:
                # negative
                if out[i][j] == 1:
                    # true negative
                    fp += 1
                else:
                    # false neg
                    tn += 1
    return tp,tn,fp,fn

def evaluation_indicator(tp,tn,fp,fn):
    try:
        tpr = float(tp) / (tp + fn)
    except ZeroDivisionError:
        tpr=0
    try:
        fpr = float(fp) / (fp + tn)
    except ZeroDivisionError:
        fpr = 0
    try:
        tnr = float(tn) / (tn + fp)
    except ZeroDivisionError:
        tnr = 0
    try:
        fnr = float(fn) / (tp + fn)
    except ZeroDivisionError:
        fnr = 0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    try:
        f1_score = 2 * p * r / (p + r)
    except ZeroDivisionError:
        f1_score = 0
    return tpr, fpr, tnr, fnr, f1_score

def reconstruct(MODEL, NODE_LIST, BASIS, ORDER):
    '''
    reconstruct(MODEL, NODE, BASIS, ORDER) returns a ranked list of the inferred
    incoming connections

     Parameters
     ------------------
     MODEL: Dynamic model employed. This is only used to specify whether the
            time series come from 1D systems like kuramoto1 or 3D systems like
            roessler. Thus, it is not used during the actual reconstruction.
     NODE:  Unit upon the reconstruction takes place. Zero indexed
     BASIS: Type of basis employed. Currently, polynomial, polynomial_diff,
            power_series, fourier, fourier_diff and RBF are supported. For
            more detailed information, please see 'Functions/basis_expansion.m'
            and Table I in the main manuscript.
     ORDER: Number of basis in the expansion.

     Input type
     ------------------
     MODEL: string
     NODE:  integer
     BASIS: string
     ORDER: integer

     Output
     ------------------
     list: Sequence of inferred interactions in the order such were detected.
     cost: Fitting cost for all inferred interactions in the order such were
           detected.
     FPR:  False positives rate for the reconstruction.
     TPR:  True positives rate for the reconstruction.
     AUC:  Quality of reconstruction measured in AUC scores.

     Example
     ------------------
     reconstruct('michaelis_menten',10,'polynomial',6) reconstructs the
     connectivity of unit 10 using polynomials up to power 6 as basis functions.

     Accompanying material to "Model-free inference of direct interactions
     from nonlinear collective dynamics".

     Author: Jose Casadiego
     Date:   May 2017
     '''

    #Stopping criterium: decrease it to recover longer list of possible links
    th = 0.00001
    print('th:' + str(th))

    models=['kuramoto1', 'kuramoto2', 'michaelis_menten', 'roessler','voter','cml']
    bases=['polynomial', 'polynomial_diff', 'fourier', 'fourier_diff', 'power_series', 'RBF']

    if (MODEL not in models):
        sys.exit('ERROR: MODEL must be a valid string: kuramoto1')

    elif (BASIS not in bases):
        sys.exit('ERROR: BASIS must be a valid string: polynomial')

    else:
        print('Initiating reconstruction...')
        print('Reading data...')
        #读取数据
        data = np.loadtxt('/data/zhangyan/Real net&dyn/gene/menten_data_600.dat', delimiter='\t')
        connectivity = np.loadtxt('/data/zhangyan/Real net&dyn/gene/menten_connectivity_600.dat', delimiter='\t')
        ts_param=np.loadtxt('/data/zhangyan/Real net&dyn/gene/menten_ts_param_600.dat', delimiter='\t')
        data=data.transpose()

        S = int(ts_param[0]) #30
        M = int(ts_param[1]) #10
        N = int(data.shape[0])  #（25，300）

        print('**************************')
        print('N:'+str(N))
        print(data.shape)
        print('**************************')

        x = data

        # Estimating time derivatives and constructing input matrices
        #估计时间导数和构造输入矩阵
        print('Estimating time derivatives and constructing input matrices...')
        Xtemp = np.array([])
        DX = np.array([])
        for s in range(S):
            m_start = M*s
            m_end = M*s + (M-1)
            x0 = x[:,m_start:m_end]
            x1 = x[:,m_start+1:m_end+1]

            Ytemp = (x0 + x1) * 0.5
            DY = x1 - x0

            #水平平铺
            Xtemp = np.hstack((Xtemp, Ytemp)) if Xtemp.size else Ytemp
            DX = np.hstack((DX, DY)) if DX.size else DY
            # print('__________________________________')
            # print(s)
            # print('Xtemp'+str(Xtemp.shape))
            # print('DX'+str(DX.shape))
            # print('__________________________________')

        X = Xtemp
        # Beginning of reconstruction algorithm
        #创建一个邻接矩阵
        predict_matrix1 = np.zeros((N,N))
        predict_matrix2 = np.zeros((N, N))
        print('Performing ARNI...')
        t_s = time.time()
        loss_dict = dict()
        mse_dict = dict()
        for NODE in NODE_LIST:

            # Y[basis, sample, node]
            Y = basis_expansion(X, ORDER, BASIS, NODE)
            #print('Y:'+str(Y.shape))
            nolist = list(range(N))
            llist = []
            cost = []
            b=1
            vec = np.zeros(N,)
            while (nolist and (b==1)):
                #composition of inferred subspaces
                Z = np.array([])
                for n in range(len(llist)):
                    Z = np.vstack((Z,Y[:,:,llist[n]])) if Z.size else Y[:,:,llist[n]]

                # projection on remaining composite spaces
                P = np.zeros((len(nolist),2))
                cost_err = np.zeros(len(nolist),)
                for n in range(len(nolist)):
                    #composition of a possible spaces
                    R = np.vstack((Z, Y[:,:,nolist[n]])) if Z.size else Y[:,:,nolist[n]]
                    #error of projection on possible composite space
                    # ( A.R=DX)
                    RI = np.linalg.pinv(R)
                    A = np.dot(DX[NODE,:], RI)
                    DX_est = np.dot(A, R)
                    DIFF = DX[NODE,:] - DX_est
                    DIFF_2 = DIFF**2
                    loss_dict[NODE] = DIFF
                    mse_dict[NODE] = DIFF_2
                    P[n,0] = np.std(DIFF) # the uniformity of error
                    P[n,1] = int(nolist[n])
                    #Fitting cost of possible composite space
                    cost_err[n] =  (1/float(M)) * np.linalg.norm(DIFF)
                    R = np.array([])

                # break if all candidates equivalent
                if np.std(P[:,0]) < th:
                    b=0
                    break

                else:
                    #Selection of composite space which minimises projection error
                    MIN = np.min(P[:,0]) #best score
                    block = np.argmin(P[:,0]) #node index of best
                    llist.append(int(P[block,1])) # add best node ID to llist
                    nolist.remove(int(P[block,1])) # remove best from candidate list
                    vec[int(P[block,1])] = MIN # used in ROC curve
                    cost.append(cost_err[block]) # record SS Error
            #print('Reconstruction has finished!')

            if not llist:
                print('WARNING: no predicted regulators - check that NODE abundance varies in the data!')
                AUC = np.nan
                FPR = [np.nan]
                TPR = [np.nan]

            else:
                for no in llist:
                    predict_matrix1[no][NODE] = vec[no]
                    predict_matrix2[no][NODE] = 1
        t_e = time.time()
        print('time for thisPerforming ARNI:' + str(round(t_e - t_s, 2)))
        print('Quality of reconstruction:')
        adjacency = connectivity
        adjacency[adjacency != 0] = 1
        # if MODEL == 'michaelis_menten':
        #     for i in range(N):
        #         adjacency[i, i] = 1

        print('--------------------------------------')
        # print(len(loss_dict.values()))
        # print(len(loss_dict))
        loss_ave = abs(sum(loss_dict.values())/len(loss_dict))
        mse_ave = abs(sum(mse_dict.values())/len(mse_dict))
        loss = np.mean(loss_ave)
        mse = np.mean(mse_ave)
        print('loss:'+str(loss))
        print('mse:' + str(mse))
        FPR1, TPR1,_ = roc_curve(adjacency.reshape(-1),
                                           predict_matrix1.reshape(-1), 1)
        # print('FPR1:'+str(FPR1))
        # print('TPR1:'+str(TPR1))
        AUC1 = auc(FPR1, TPR1)
        print('AUC1:'+str(AUC1))
        print('--------------------------------------')

        print('--------------------------------------')
        FPR2, TPR2, _ = roc_curve(adjacency.reshape(-1),
                                  predict_matrix2.reshape(-1), 1)
        # print('FPR2:' + str(FPR2))
        # print('TPR2:' + str(TPR2))
        AUC2 = auc(FPR2, TPR2)
        print('AUC2:' + str(AUC2))
        print('--------------------------------------')

        # 算err
        err = np.sum(np.abs(predict_matrix2 - adjacency))
        # 计算tp,,
        tp, tn, fp, fn = calc_tptnfpfn(predict_matrix2, adjacency)
        tpr, fpr, tnr, fnr, f1_score = evaluation_indicator(tp, tn, fp, fn)
        print('err:', err)
        print('tp:', tp)
        print('tn:', tn)
        print('fp:', fp)
        print('fn:', fn)
        print('tpr:', tpr)
        print('fpr:', fpr)
        print('tnr:', tnr)
        print('fnr:', fnr)
        print('f1:', f1_score)

        print('-----------------------------------------')

    return(predict_matrix2, cost, FPR1, TPR1, AUC1, FPR2, TPR2, AUC2)

if __name__ =='__main__':
    ORDER = 6
    BASIS = ('polynomial', 'polynomial_diff', 'fourier', 'fourier_diff', 'power_series', 'RBF')
    NAMES = ('Polynomial', 'Polynomial Diff', 'Fourier', 'Fourier Diff', 'Power Series', 'Radial Basis Function')
    # BASIS = ('polynomial', 'fourier', 'power_series', 'RBF')
    # NAMES = ('Polynomial', 'Fourier', 'Power Series', 'Radial Basis Function')
    NODE_LIST = list(range(100))

    f1, axes1 = plt.subplots(2, 3)
    axes1 = np.ravel(axes1)

    for i in range(len(BASIS)):
        print('BASIS:'+BASIS[i])
        predict_matrix2, cost, FPR1, TPR1, AUC1, FPR2, TPR2, AUC2 = reconstruct('michaelis_menten',NODE_LIST, BASIS[i], ORDER)
        axes1[i].plot(FPR1, TPR1)
        axes1[i].set_xlabel('FPR1')
        axes1[i].set_ylabel('TPR1')
        axes1[i].set_title( NAMES[i]+'AUC1 score=%.3f' % AUC1, fontsize=8)

    f1.tight_layout()
    f1.subplots_adjust(top=0.80)


plt.show()

end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('end_time:', end_time)