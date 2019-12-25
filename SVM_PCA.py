#!/usr/bin/env python 
# encoding: utf-8 

"""
@version: v1.0
@author: Shirley
@site: Southeast University
@software: PyCharm
@file: load_file.py
@time: 2019/12/4 15:32
"""

# load libarary
import os.path
from scipy.io import loadmat
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import load_data # 可以换成地下注释掉的部分
# # load data
# def load_data(path,label):
#     os.chdir(path)
#     files = os.listdir(path)
#     n = len(files)
#     com1 = np.zeros([1, 8100])
#     for i in range(0, n):
#         Data_dict = loadmat(files[i])
#         Data_116 = Data_dict['con_mat_FA']
#         Data_90 = Data_116[0:90, 0:90]  # 90*90 FA矩阵
#         Data_1 = np.reshape(Data_90, (1, 8100))
#         com1 = np.row_stack((com1, Data_1))
#     # Train set data
#     Data = com1[1:n+1, 0:8100]
#     # 标签y
#     y = np.array(label).transpose()
#     return Data, y

# 读入训练集数据
path = 'E:\\Experiment2\\train'
label = [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1,
         1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1,
         1, 0, 0, 0]
X,y=load_data.load_data(path,label)

## reduce_dimension
# PCA降维
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def scatter_PCA(X,label):
    plt.figure()
    pca=PCA(n_components=2)
    reduced_x=pca.fit_transform(X)
    plt.scatter(reduced_x[:,0],reduced_x[:,1],s=40, c=label,cmap=plt.cm.Spectral)
    plt.show()
# scatter_PCA(X,y)

# TSNE降维
from sklearn.manifold import TSNE
def scatter_TSNE(X,label):
    tsne=TSNE(n_components=2, verbose=1, perplexity=40,n_iter=300)
    reduced_x=tsne.fit_transform(X)
    plt.scatter(reduced_x[:,0],reduced_x[:,1],s=40, c=label,cmap=plt.cm.Spectral)
    plt.show()
# scatter_TSNE(X,y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
# random_state的值相当于一种规则，通过设定为相同的数，每次分割的结果都是相同的，如果设置为默认的话，每次得到的分割结果都不一样

from sklearn import svm
from sklearn.metrics import confusion_matrix

def get_accuracy_score(n,X_train, X_test, y_train, y_test):
    '''当主成分为n时,计算模型预测的准确率'''
    pca = PCA(n_components = n)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    # 使用支持向量机分类器
    clf = svm.SVC()
    clf.fit(X_train_pca, y_train)
    # 计算准确度
    accuracy = clf.score(X_test_pca, y_test)
    y_pred = clf.predict(X_test_pca)
    print('n_components:{:.2f} , accuracy:{:.4f}'.format(n, accuracy))
    print(y_pred)
    print(confusion_matrix(y_test, y_pred))
    return accuracy
n=4
score = get_accuracy_score(n, X_train, X_test, y_train, y_test)







