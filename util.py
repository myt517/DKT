import itertools
import subprocess
import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from sklearn.decomposition import PCA
import torch
import copy
import torch.nn.functional as F
import random
import csv
import sys
from torch import nn
from tqdm import tqdm_notebook, trange, tqdm
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME,CONFIG_NAME,BertPreTrainedModel,BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.manifold import TSNE
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple
matplotlib.use('Agg')

def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc

def clustering_score(y_true, y_pred):
    return {'ACC': round(clustering_accuracy_score(y_true, y_pred)*100, 2),
            'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 2),
            'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2)}


def pca_visualization(X: np.ndarray,
                      y: pd.Series,
                      classes: List[int],
                      save_path: str):
    """
    Apply PCA visualization for features.
    """
    red_features = PCA(n_components=2, svd_solver="full").fit_transform(X)

    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots()
    for _class in classes:
        if _class == "unseen":
            ax.scatter(red_features[y == _class, 0], red_features[y == _class, 1],
                    label=_class, alpha=0.5, s=20, edgecolors='none', color="gray")
        else:
            ax.scatter(red_features[y == _class, 0], red_features[y == _class, 1],
                    label=_class, alpha=0.5, s=20, edgecolors='none', zorder=10)
    ax.legend()
    ax.grid(True)
    plt.savefig(save_path, format="png")


def TSNE_visualization(X: np.ndarray,
                      y: pd.Series,
                      classes: List[str],
                      save_path: str):
    X_embedded = TSNE(n_components=2).fit_transform(X)

    color_list=["red","green","blue","yellow","purple","black","brown","cyan","gray","pink","orange","blueviolet","greenyellow","sandybrown","deeppink"]

    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots()
    for _class in classes:
        if _class == "unseen":
            ax.scatter(X_embedded[y == _class, 0], X_embedded[y == _class, 1],
                       label=_class, alpha=0.5, s=20, edgecolors='none', color="gray")
        else:
            ax.scatter(X_embedded[y == _class, 0], X_embedded[y == _class, 1],
                       label=_class, alpha=0.5, s=4, edgecolors='none', zorder=15, color=color_list[_class])

    ax.grid(True)
    #plt.savefig(save_path, bbox_inches='tight', pad_inches=0, format="pdf")
    plt.savefig(save_path, format="png")

    print()

def Line_chart():

    epoch1 = [1, 0.8, 0.6, 0.4, 0.2]
    model1 = [97.78, 95.56,	97.33, 97.33, 93.33]
    model2 = [95.7,	94.22,	94.22,	93.78,	92.44]
    model3 = [70.22, 60.44,	64.89, 56.44, 66.22]
    '''
    epoch1 = [1, 0.8, 0.6, 0.4, 0.2]
    model1 = [97.78, 96, 97.78, 96, 97.78]
    model2 = [95.7, 96, 95.11, 95.11, 92.89]
    model3 = [70.22, 60, 57.33, 59.11, 54.67]
    '''
    # sns.set(style="darkgrid")
    plt.plot(epoch1, model1, linewidth=2, c="#41719c", marker=".", ms=9, label="DKT")
    plt.plot(epoch1, model2, linewidth=2, c="#70ad47", marker=".", ms=9, label="DeepAligned")
    plt.plot(epoch1, model3, linewidth=2, c="#ed7d31", marker=".", ms=9, label="PTK-means")
    plt.yticks([20, 40, 60, 80, 100], fontsize=15)
    plt.xticks([1, 0.8, 0.6, 0.4, 0.2], fontsize=15)
    plt.ylim([50, 100])
    plt.xlim([1.1, 0.1])

    plt.ylabel("ACC", fontsize=15)
    plt.xlabel("(b) Ratio of IND sample numbers", fontsize=15)
    plt.legend(loc='center right', fontsize=15)
    plt.savefig("./outputs/epoch.png", bbox_inches='tight', pad_inches=0.05)
    plt.savefig("./outputs/sample.pdf", bbox_inches='tight', pad_inches=0.025)


def intra_distance(X, predicted_y, num_labels):
    cluster_center = []
    for i in range(num_labels):
        X_feats = X[predicted_y == i]
        center_x = np.mean(X_feats, axis=0)
        cluster_center.append(center_x)
    #print(len(cluster_center))

    intra_cluster_distance = []
    for i in range(num_labels):
        X_feats = X[predicted_y == i]
        #dist = np.dot(X_feats, cluster_center[i].T)
        dist = np.sqrt(np.sum(np.square(X_feats - cluster_center[i]), axis=1))
        dist = dist.tolist()
        intra_cluster_distance.append(dist)

    min_list, max_list, mean_list = [], [], []
    for i in range(len(intra_cluster_distance)):
        min_list.append(min(intra_cluster_distance[i]))
        max_list.append(max(intra_cluster_distance[i]))
        mean_list.append(sum(intra_cluster_distance[i])/len(intra_cluster_distance[i]))


    min_d = min(min_list)
    max_d = max(max_list)
    mean_d = sum(mean_list)/len(mean_list)

    return min_d, max_d, mean_d, mean_list


def inter_distance(X, predicted_y, num_labels):
    cluster_center = []
    for i in range(num_labels):
        X_feats = X[predicted_y == i]
        #print(X_feats)
        center_x = np.mean(X_feats, axis=0)
        cluster_center.append(center_x)
    print(len(cluster_center), cluster_center[0].shape)

    #print(cluster_center)
    #print("______________________________________")

    #knn.fit(cluster_center, labels)
    #y_pred = knn.predict(cluster_center)
    #print(y_pred)

    inter_cluster_distance = []
    for i in range(len(cluster_center)):
        dist_list = []
        for j in range(len(cluster_center)):
            #dist = np.dot(cluster_center[i], cluster_center[j].T)
            dist = np.sqrt(np.sum(np.square(cluster_center[i] - cluster_center[j])))
            dist_list.append(dist)
        inter_cluster_distance.append(dist_list)

    inter_distance = []
    for i in range(len(inter_cluster_distance)):
        inter_cluster_distance[i].sort()
        #print(inter_cluster_distance[i])
        tmp = np.mean(inter_cluster_distance[i][-4:-1])
        inter_distance.append(tmp)

    inter_distance = np.array(inter_distance)
    print(inter_distance.shape)

    # Min
    min_d = np.float(inter_distance.min())
    # Max
    max_d = np.float(inter_distance.max())
    # Mean
    mean_d = np.float(inter_distance.mean())


    return min_d, max_d, mean_d, inter_distance


def select_hard_2(intra_list, inter_list):
    a = []
    for i in range(23):
        a.append(inter_list[i]/intra_list[i])
    return a

def select_hard(y_true, y_pred):
    acc_list = []

    for i in range(23):
        y_true_selected = y_true[y_true==i]
        y_pred_selected = y_pred[y_true==i]
        results = clustering_score(y_true_selected, y_pred_selected)
        acc_list.append(results["ACC"])

    return acc_list


if __name__ == '__main__':

    print('Data and Parameters Initialization...')
    Line_chart()