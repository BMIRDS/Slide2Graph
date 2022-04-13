import numpy as np
import os
from pathlib import Path
import scipy.spatial as spt
import re
from torch_geometric.data import Data
from config import Config
import torch
import networkx as nx
import torch_geometric.utils as gutils
import matplotlib.pyplot as plt
import torch_geometric
import torch_geometric.data as gdata
from torch_geometric.data import DataLoader as gDataLoader
from torch_scatter import scatter_mean
from config import Config
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score





config = Config()




def generate_data(raw_array,label_array):
    """
    Generate graph data without whole slide info from raw data
    Args:
        raw_array: a matrix of patches' features extracted by ResNet
        label_array: a matrix of patches' labels
    """
    candidate_position = np.nonzero(raw_array)
    pos_x, pos_y = filter_positions(candidate_position)
    num_of_nodes = len(pos_x)
    points = [ [pos_x[i], pos_y[i]] for i in range(num_of_nodes)]
    nodes = np.array([raw_array[points[i][0]][points[i][1]] for i in range(len(points))])
    node_labels = np.array([label_array[points[i][0]][points[i][1]] for i in range(len(points))])
    node_labels = torch.from_numpy(node_labels).long()
    pos = np.array([[points[i][0],points[i][1]] for i in range(len(points))])  
    pos = torch.from_numpy(pos)
    if len(nodes.shape) < 2:
        nodes = np.expand_dims(nodes, axis=1)
    nodes = torch.from_numpy(nodes)
    edges = []
    attrs = []
    label = torch.from_numpy(np.array(int(re.findall("_(.*?).npy", item)[0])).reshape(1))
    ckt = spt.cKDTree(points) 
    for point_idx in range(len(points)):
        d, x = ckt.query(points[point_idx],k=4)  
        d = d[1:] 
        x = x[1:]
        min_dis_index = d==min(d)
        d = d[min_dis_index]  
        x = x[min_dis_index] 
        if min(d) == 1:
            edges.extend([[x_idx,point_idx] for x_idx in x])
            attrs.extend([(1/dist) for dist in d])
        edges.extend([[point_idx,x_idx] for x_idx in x])
        attrs.extend([1/dist for dist in d])
    edges = torch.from_numpy(np.transpose(np.array(edges)))
    attrs = torch.from_numpy(np.array(attrs).reshape(-1,1))
    print(f'original matrix shape: {raw_array.shape}')
    print(nodes.shape)  
    print(edges.shape) 
    print(attrs.shape) 
    print(label.shape)
    a_graph = Data(x=nodes,edge_index=edges,edge_attr=attrs,y=node_labels,pos=pos,graph_y=label)
    return a_graph


def filter_positions(candidate_position):
    """
    Remove some redundant positions
    Args:
        candidate_position: all possible positions
    """
    cdd_position_set = set()
    x,y,z = candidate_position
    for item in zip(x,y):
        cdd_position_set.add(item)
    cdd_position_set = list(cdd_position_set)
    new_x = []
    new_y = []
    for item_pos in cdd_position_set:
        new_x.append(item_pos[0])
        new_y.append(item_pos[1])
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    return new_x, new_y


def plot_graph(a_graph):
    """
    plot the topology structure of a graph
    Args:
        a_graph: the graph need to be visualized
    """
    g_nx = gutils.to_networkx(a_graph,to_undirected=True)
    labeldict = {list(g_nx.nodes())[i]:str(a_graph.y[i].numpy()) for i in range(len(a_graph.y))}
    nx.draw_kamada_kawai(g_nx,labels=labeldict,with_labels=True)
    print(str(a_graph.graph_y.numpy()))
    plt.show()



def get_dataset(raw_array,label_array,wsi_id,wsi_label):
    """
    Generate graph data from raw data
    Args:
        raw_array: a matrix of patches' features extracted by ResNet
        label_array: a matrix of patches' labels
        wsi_id: Whole Slide Image ID
        wsi_label: Whole Slide Image Label
    """
    candidate_position = np.nonzero(raw_array) 
    pos_x, pos_y = filter_positions(candidate_position)
    num_of_nodes = len(pos_x)
    points = [ [pos_x[i], pos_y[i]] for i in range(num_of_nodes)]
    nodes = np.array([raw_array[points[i][0]][points[i][1]] for i in range(len(points))])
    node_labels = np.array([label_array[points[i][0]][points[i][1]] for i in range(len(points))])
    node_labels = torch.from_numpy(node_labels).long()
    pos = np.array([[points[i][0],points[i][1]] for i in range(len(points))])  
    pos = torch.from_numpy(pos)
    if len(nodes.shape) < 2:
        nodes = np.expand_dims(nodes, axis=1)
    nodes = torch.from_numpy(nodes)
    edges = []
    attrs = []
    label = torch.from_numpy(np.array(wsi_label).reshape(1))
    slide_idx = wsi_id
    ckt = spt.cKDTree(points) 
    for point_idx in range(len(points)):
        d, x = ckt.query(points[point_idx],k=5)  
        d = d[1:] 
        x = x[1:]
        min_dis_index = d==min(d)
        edges.extend([[point_idx,x_idx] for x_idx in x])
        attrs.extend([(1/dist) for dist in d])
    edges = torch.from_numpy(np.transpose(np.array(edges)))
    attrs = torch.from_numpy(np.array(attrs).reshape(-1,1))
    a_graph = Data(x=nodes.float(),edge_index=edges,edge_attr=attrs.squeeze(1).float(),y=node_labels,pos=pos,graph_y=label,slide_index=slide_idx)
    return a_graph





def boots_ci(y_true,y_pred,bs_times=10000):
    """
    Calculate confidence interval using bootstrap
    Args:
        y_true: Labels
        y_pred: Predictions
        bs_times: sample times
    """
    alpha = 5.0
    num_observations = len(y_true)
    np.random.seed(1)
    metrics_list = [0,1,2]
    output_list = []
    for metrc in metrics_list:
        f_score_list = []
        for i in range(bs_times):
            indices = resample(range(num_observations),n_samples=num_observations)
            bs_true = y_true[indices]
            bs_pred = y_pred[indices]
            f_score_list.append(score(bs_true, bs_pred)[metrc])
        f_scores = np.array(f_score_list)
        avg = np.median(f_scores,axis=0)
        lower_p = alpha / 2.0
        upper_p = (100 - alpha) + (alpha / 2.0)
        lower = np.percentile(f_scores, lower_p,axis=0)
        upper = np.percentile(f_scores, upper_p,axis=0)
        output_list.append([avg, lower, upper])
    return output_list

def auc_ci(y_true,y_score,bs_times=10000):
    """
    Calculate confidence interval of AUC using boostrap
    Args:
        y_true: Labels
        y_score: Probabilities 
        bs_times: sample times
    """
    alpha = 5.0
    num_observations = len(y_true)
    np.random.seed(1)
    output_list = []
    num_cls = y_score.shape[1]
    for a_cls in range(num_cls):
        f_score_list = []
        for i in range(bs_times):
            indices = resample(range(num_observations),n_samples=num_observations)
            bs_true = y_true[indices]
            bs_score = y_score[indices,a_cls]
            fpr, tpr, thresholds = metrics.roc_curve([item for item in bs_true], bs_score, pos_label=a_cls)
            f_score_list.append(metrics.auc(fpr, tpr))
        f_score_list = [item for item in f_score_list]
        f_scores = np.array(f_score_list)
        lower_p = alpha / 2.0
        upper_p = (100 - alpha) + (alpha / 2.0)
        fpr, tpr, thresholds = metrics.roc_curve([item for item in y_true], y_score[:,a_cls], pos_label=a_cls)
        avg = metrics.auc(fpr,tpr)
        lower = np.percentile(f_scores, lower_p,axis=0)
        upper = np.percentile(f_scores, upper_p,axis=0)
        output_list.append([avg, lower, upper])
    return output_list


def boots_fscores(y_true,y_pred,bs_times=10000):
    """
    Calculate confidence interval of f scores using boostrap
    Args:
        y_true: Labels
        y_pred: Predictions
        bs_times: sample times
    """
    num_observations = len(y_true)
    np.random.seed(1)
    f_neo_list = []
    f_neg_list = []
    f_pos_list = []
    f_all_list = []
    for i in range(bs_times):
        indices = resample(range(num_observations),n_samples=num_observations)
        bs_true = y_true[indices]
        bs_pred = y_pred[indices]
        f_neo_list.append(score(bs_true, bs_pred)[2][0])
        f_neg_list.append(score(bs_true, bs_pred)[2][1])
        f_pos_list.append(score(bs_true, bs_pred)[2][2])
        f_all_list.append(f1_score(bs_true,bs_pred,average='macro'))
    f_scores = [f_neo_list,f_neg_list,f_pos_list,f_all_list]
    return f_scores
