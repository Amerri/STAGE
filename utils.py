from matplotlib.pyplot import delaxes
import torch
import random
import os
import os.path as osp
import pandas as pd
import numpy as np
import torch.nn as nn
import time
import scipy.sparse as sparse
import dgl as dl
import torch.utils.data as D
from dgl.data.utils import load_graphs
from torch.utils.data import Subset
import sklearn.preprocessing as preprocessing
from scipy.sparse import linalg
import torch.nn.functional as F


class HyperG:
    def __init__(self, H, X=None, w=None):
        """ Initial the incident matrix, node feature matrix and hyperedge weight vector of hypergraph
        :param H: scipy coo_matrix of shape (n_nodes, n_edges)
        :param X: numpy array of shape (n_nodes, n_features)
        :param w: numpy array of shape (n_edges,)
        """
        assert sparse.issparse(H)
        assert H.ndim == 2

        self._H = H
        self._n_nodes = self._H.shape[0]
        self._n_edges = self._H.shape[1]

        if X is not None:
            assert isinstance(X, np.ndarray) and X.ndim == 2
            self._X = X
        else:
            self._X = None

        if w is not None:
            self.w = w.reshape(-1)
            assert self.w.shape[0] == self._n_edges
        else:
            self.w = np.ones(self._n_edges)

        self._DE = None
        self._DV = None
        self._INVDE = None
        self._DV2 = None
        self._THETA = None
        self._L = None

    def num_edges(self):
        return self._n_edges

    def num_nodes(self):
        return self._n_nodes

    def incident_matrix(self):
        return self._H

    def hyperedge_weights(self):
        return self.w

    def node_features(self):
        return self._X

    def node_degrees(self):
        if self._DV is None:
            H = self._H.tocsr()
            dv = H.dot(self.w.reshape(-1, 1)).reshape(-1)
            self._DV = sparse.diags(dv, shape=(self._n_nodes, self._n_nodes))
        return self._DV

    def edge_degrees(self):
        if self._DE is None:
            H = self._H.tocsr()
            de = H.sum(axis=0).A.reshape(-1)
            self._DE = sparse.diags(de, shape=(self._n_edges, self._n_edges))
        return self._DE

    def inv_edge_degrees(self):
        if self._INVDE is None:
            self.edge_degrees()
            inv_de = np.power(self._DE.data.reshape(-1), -1.)
            self._INVDE = sparse.diags(inv_de, shape=(self._n_edges, self._n_edges))
        return self._INVDE

    def inv_square_node_degrees(self):
        if self._DV2 is None:
            self.node_degrees()
            dv2 = np.power(self._DV.data.reshape(-1)+1e-6, -0.5)
            self._DV2 = sparse.diags(dv2, shape=(self._n_nodes, self._n_nodes))
        return self._DV2

    def theta_matrix(self):
        if self._THETA is None:
            self.inv_square_node_degrees()
            self.inv_edge_degrees()

            W = sparse.diags(self.w)
            self._THETA = self._DV2.dot(self._H).dot(W).dot(self._INVDE).dot(self._H.T).dot(self._DV2)

        return self._THETA

    def laplacian(self):
        if self._L is None:
            self.theta_matrix()
            self._L = sparse.eye(self._n_nodes) - self._THETA
        return self._L

    def update_hyedge_weights(self, w):
        assert isinstance(w, (np.ndarray, list)), \
            "The hyperedge array should be a numpy.ndarray or list"

        self.w = np.array(w).reshape(-1)
        assert w.shape[0] == self._n_edges

        self._DV = None
        self._DV2 = None
        self._THETA = None
        self._L = None

    def update_incident_matrix(self, H):
        assert sparse.issparse(H)
        assert H.ndim == 2
        assert H.shape[0] == self._n_nodes
        assert H.shape[1] == self._n_edges

        # TODO: reset hyperedge weights?

        self._H = H
        self._DE = None
        self._DV = None
        self._INVDE = None
        self._DV2 = None
        self._THETA = None
        self._L = None



class Classifier(nn.Module):
    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.linear = nn.Linear(n_hid, n_out)

    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)

    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)

# def load_hete_graph():
#     rel_bc2bc = np.load('data/relation_invest_bc2bc.npy')
#     fin_feature_ttl = np.load('data/features_norm.npy')
#     src, tgt = zip(*rel_bc2bc)

#     graph_data = { ('company','invest_bc2bc','company') : (src,tgt) }

#     g = dl.heterograph(graph_data)

#     num = g.num_nodes()

#     feats = torch.tensor(fin_feature_ttl[:num])

#     g.nodes['company'].data['feature'] = feats.float()

#     dict_node_feats =  {'company': feats.float()}

#     return g, feats, dict_node_feats  

from sklearn.ensemble import RandomForestClassifier

def load_hete_graph():
    rel_bc2bc = np.load('data/listed_comp/relation_invest_bc2bc.npy')
    rel_provide_bc2bc = np.load('data/listed_comp/relation_provide_bc2bc.npy')
    rel_sale_bc2bc = np.load('data/listed_comp/relation_sale_bc2bc.npy')

    feature_fin = np.load('data/listed_comp/features_norm_bc.npy')
    feature_basic = np.load('data/listed_comp/features_basic_bc.npy')

    # feature_basic1 = np.nan_to_num(feature_basic)
    # print(feature_basic1.shape)
    # fin_feature_ttl = np.load('data/features_100.npy')

    src, tgt = zip(*rel_bc2bc)
    src_p,tgt_p = zip(*rel_provide_bc2bc)
    src_s,tgt_s = zip(*rel_sale_bc2bc)

    graph_data = { ('company','invest_bc2bc','company') : (src,tgt),
                   ('company','provide_bc2bc','company') : (src_p,tgt_p),
                   ('company','sale_bc2bc','company') : (src_s,tgt_s)}

    g = dl.heterograph(graph_data)

    num = g.num_nodes()


    feats_fin = torch.tensor(feature_fin[:num])
    feats_bsc = torch.tensor(feature_basic[:num])
    feats = torch.cat([feats_fin,feats_bsc],1)

    
    g.nodes['company'].data['feature'] = feats.float()

    dict_node_feats =  {'company': feats.float()}

    return g, g.nodes['company'].data['feature'], dict_node_feats


def feature_importance(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    importances = model.feature_importances_
    return importances

def load_hete_graph2():
    rel_bc2bc = np.load('data/listed_comp/relation_invest_bc2bc.npy')
    rel_provide_bc2bc = np.load('data/listed_comp/relation_provide_bc2bc.npy')
    rel_sale_bc2bc = np.load('data/listed_comp/relation_sale_bc2bc.npy')

    feature_fin = np.load('data/listed_comp/features_norm_bc.npy')
    feature_basic = np.load('data/listed_comp/features_basic_bc.npy')

    # feature_basic1 = np.nan_to_num(feature_basic)
    # print(feature_basic1.shape)
    # fin_feature_ttl = np.load('data/features_100.npy')

    src, tgt = zip(*rel_bc2bc)
    src_p,tgt_p = zip(*rel_provide_bc2bc)
    src_s,tgt_s = zip(*rel_sale_bc2bc)

    graph_data = { ('company','invest_bc2bc','company') : (src,tgt),
                   ('company','provide_bc2bc','company') : (src_p,tgt_p),
                   ('company','sale_bc2bc','company') : (src_s,tgt_s)}

    g = dl.heterograph(graph_data)

    num = g.num_nodes()


    feats_fin = torch.tensor(feature_fin[:num])
    feats_bsc = torch.tensor(feature_basic[:num])
    feats = torch.cat([feats_fin,feats_bsc],1)

    labels = np.load('data/listed_comp/risk_bc_label_4.npy')
    # 计算特征重要性
    importances = feature_importance(feats, labels)

    threshold = 0.02
    x_selected = feats[:, importances > threshold]

    # print(feats)

    # feats = torch.from_numpy(np.nan_to_num(feats))

    # isnan = np.isnan(feats)
    # print("空值：",True in isnan)
    
    g.nodes['company'].data['feature'] = x_selected.float()

    dict_node_feats =  {'company': x_selected.float()}

    return g, x_selected, dict_node_feats

def load_hete_graph3():


    device=torch.device('cuda')

    g =  dl.load_graphs("data/lst_comps7.dgl")[0][0].to('cuda:0')


    feats = g.nodes['company'].data['feature'] 

    g.nodes['company'].data['feature'] = feats.float().to(device)

    dict_node_feats =  {'company': feats.float().to(device)}

    return g, g.nodes['company'].data['feature'], dict_node_feats

def load_hete_graph4():

    g = dl.load_graphs('data/lst_comps7.dgl')[0][0]

    feats = g.nodes['company'].data['feature'] 

    labels = np.load('data/listed_comp/risk_bc_label_4.npy')
    # 计算特征重要性
    importances = feature_importance(feats, labels)

    threshold = 0.02
    x_selected = feats[:, importances > threshold]

    g.nodes['company'].data['feature'] = x_selected.float()

    dict_node_feats =  {'company': feats.float()}

    return g, x_selected, dict_node_feats



def  set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def gen_attribute_hg(n_nodes, attr_dict, X=None):
    # 构建超图集合的子图
    """
    :param attr_dict: dict, eg. {'attri_1': [node_idx_1, node_idx_1, ...], 'attri_2':[...]} (zero-based indexing)
    :param n_nodes: int,
    :param X: numpy array, shape = (n_samples, n_features) (optional)
    :return: instance of HyperG
    """

    if X is not None:
        assert n_nodes == X.shape[0]

    n_edges = len(attr_dict)
    node_idx = []
    edge_idx = []

    for idx, attr in enumerate(attr_dict):
        nodes = sorted(attr_dict[attr])
        node_idx.extend(nodes)
        edge_idx.extend([idx] * len(nodes))

    node_idx = np.asarray(node_idx)
    edge_idx = np.asarray(edge_idx)
    values = np.ones(node_idx.shape[0])

    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))
    return HyperG(H, X=X)


def split_data():
    g, feats, dict_node_features = load_hete_graph()
    # labels_ttl = np.load('data/risk_label.npy')
    labels_ttl = np.load('data/listed_comp/risk_bc_label_4.npy')
    num_nodes = g.num_nodes()
    labels = torch.tensor(labels_ttl[:num_nodes])
    train_size = int(num_nodes * 0.6)
    val_size = int(num_nodes * 0.2)
    test_size = num_nodes - train_size - val_size
    train_data, val_data, test_data = torch.utils.data.random_split(feats,[train_size,val_size,test_size])
    return train_data, val_data, test_data

def my_load_data():
    g, features, _ = load_hete_graph3()
    train_mask = g.ndata['train_mask']
    train_indices = torch.nonzero(train_mask).squeeze()
    train_data = Subset(features, train_indices)

    valid_mask = g.ndata['valid_mask']
    valid_indices = torch.nonzero(valid_mask).squeeze()
    valid_data = Subset(features, valid_indices)

    test_mask = g.ndata['test_mask']
    test_indices = torch.nonzero(test_mask).squeeze()
    test_data = Subset(features, test_indices)

    return train_data, valid_data, test_data

def my_split_data(train_idx, val_idx, test_idx):
    _, features, _ = load_hete_graph3()
    train_data = Subset(features, train_idx)
    val_data = Subset(features, val_idx)
    test_data = Subset(features, test_idx)
    return train_data, val_data, test_data

import json

def load_hyper_graph():
    filename = 'data/dicts_hyper.json'
    with open(filename,"r") as f: 
        hyper_graph = json.load(f)
    return hyper_graph

# def load_train_hyper_graph(train_data):
#     hyper_graph = load_hyper_graph()
#     train_idx = train_data.indices
#     dicts_industry = hyper_graph['industry']
#     dicts_train = { _key :[] for _key in dicts_industry}
#     for idx in train_idx:
#         for key in dicts_industry:
#             value = dicts_industry[key]
#             if idx in value:
#                 dicts_train[key].append(idx)
#     return dicts_train

def load_sub_hyper_graph(hyper_graph_data): # hyper_graph_data : dict
    hyper_graph = load_hyper_graph()
    if type(hyper_graph_data.indices) == torch.Tensor:
        hyper_graph_data.indices = hyper_graph_data.indices.tolist()
    train_idx = hyper_graph_data.indices
    dicts_industry = hyper_graph['industry']
    dicts_sub_hyper_graph = { _key :[] for _key in dicts_industry}
    for idx in train_idx:
        for key in dicts_industry:
            value = dicts_industry[key]
            if idx in value:
                dicts_sub_hyper_graph[key].append(idx)
    return {'industry':dicts_sub_hyper_graph}


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy的sparse matrix转换成torch的sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x

def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    n = g.number_of_nodes()
    # adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float)
    adj = g.adj().astype(float)
    norm = sparse.diags(
        dl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
    )
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata["pos_undirected"] = x.float()
    return g
