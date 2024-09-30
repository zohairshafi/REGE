import torch
import scipy 
import math
import random
import GraphRicciCurvature

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import torch.nn.functional as F
import scipy.sparse as sp
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
from copy import deepcopy

from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
from tqdm import tqdm
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.utils import negative_sampling, convert, to_dense_adj
from torch.utils.data import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from scipy.stats import entropy

from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN, RGCN, SimPGCN
from deeprobust.graph.global_attack import Metattack, DICE
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import PrePtbDataset
from deeprobust.graph import utils

from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import numpy as np
import torch
import os
import igraph

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def largest_connected_components(adj, n_components=1):
    """Select the largest connected components in the graph.
    Parameters
    ----------
    adj : gust.SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.
    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep


    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep

def load_npz(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                    loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                         loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')

    return adj_matrix, attr_matrix, labels

def preprocess_graph(adj, I=True):
    """
    Perform the processing of the adjacency matrix proposed by Kipf et al. 2017.

    Parameters
    ----------
    adj: sp.spmatrix
        Input adjacency matrix.

    Returns
    -------
    The matrix (D+1)^(-0.5) (adj + I) (D+1)^(-0.5)

    """
    if I:
        adj_ = adj + sp.eye(adj.shape[0])
    else:
        adj_ = adj
    rowsum = adj_.sum(1).A1
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
    return adj_normalized

def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None,
                                 random_state=None):

    """
    Split the arrays or matrices into random train, validation and test subsets.
    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices.
    train_size : float, default 0.5
        Proportion of the dataset included in the train split.
    val_size : float, default 0.3
        Proportion of the dataset included in the validation split.
    test_size : float, default 0.2
        Proportion of the dataset included in the test split.
    stratify : array-like or None, default None
        If not None, data is split in a stratified fashion, using this as the class labels.
    random_state : int or None, default None
        Random_state is the seed used by the random number generator;
    Returns
    -------
    splitting : list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.
    """
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def load_data(name='cora_0.1edges_Meta-Self.npy', dataset='cora.npz', data_dir = '/home/shafi.z/GraphDiffusion/data/', attack=True):

    _A_obs, _X_obs, _z_obs = load_npz(data_dir +dataset)
    if _X_obs is None:
        _X_obs = sp.eye(_A_obs.shape[0]).tocsr()

    lcc = largest_connected_components(_A_obs)

    if attack:
        try:
            _A_obs = sp.csr_matrix(np.load(data_dir+name))
        except:
            _A_obs = sp.csr_matrix(np.array(np.load(data_dir +name), dtype = int))
        
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    if not attack:
        _A_obs = _A_obs[lcc][:,lcc]
    _A_obs.setdiag(0)
    _A_obs = _A_obs.astype("float32")
    _A_obs.eliminate_zeros()
    _X_obs = _X_obs.astype("float32")

    assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
    assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"

    _X_obs = _X_obs[lcc]
    _z_obs = _z_obs[lcc]
    _N = _A_obs.shape[0]
    _K = _z_obs.max()+1
    _Z_obs = np.eye(_K)[_z_obs]

    seed = 15
    unlabeled_share = 0.8
    val_share = 0.1
    train_share = 1 - unlabeled_share - val_share
    np.random.seed(seed)

    split_train, split_val, split_unlabeled = train_val_test_split_tabular(np.arange(_N),
                                                                           train_size=train_share,
                                                                           val_size=val_share,
                                                                           test_size=unlabeled_share,
                                                                           stratify=_z_obs)
    split_unlabeled = np.union1d(split_val, split_unlabeled)
    
    labels = torch.LongTensor(_Z_obs)
    labels = torch.max(labels, dim=1)[1]
    features = torch.FloatTensor(np.array(_X_obs.todense())).float()
    
    args_cuda = torch.cuda.is_available()
    set_seed(42, args_cuda)
    if args_cuda:
        labels = labels.cuda()
        features = features.cuda()
    
    return labels, features, split_train, split_val, split_unlabeled, _A_obs

def plot_with_labels(embedding, labels, dataset, phase):

    tsne = TSNE(n_components=2)
    lowDWeights = tsne.fit_transform(embedding)
    cc = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    plt.cla()
    plt.rcParams['figure.dpi'] = 300
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    i = 0
    for x, y, s in zip(X, Y, labels):
        plt.scatter(x, y, 2, c=cc[s])
        i += 1
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.axis('off')
    plt.title(phase + ' Graph ' + '(' + dataset + ')')
    plt.savefig('result/' + phase + '_' + dataset + '.png', dpi=300)
    plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def largest_connected_component(graph):
    # Get a list of all connected components, sorted by size
    connected_components = sorted(nx.connected_components(graph), key=len, reverse=True)

    # The largest connected component is the first in the list
    largest_connected_component = graph.subgraph(connected_components[0])

    return largest_connected_component

def probabilistic_round(x):
    return int(math.floor(x + random.random()))

def show_results(G, curvature="ricciCurvature"):

    # Print the first five results
    for n1,n2 in list(G.edges())[:5]:
        print("Ricci curvature of edge (%s,%s) is %f" % (n1 ,n2, G[n1][n2][curvature]))

    # Plot the histogram of Ricci curvatures
    plt.subplot(2, 1, 1)
    ricci_curvtures = nx.get_edge_attributes(G, curvature).values()
    plt.hist(ricci_curvtures,bins=20)
    plt.xlabel('Ricci curvature')
    plt.title("Histogram of Ricci Curvatures")

    # Plot the histogram of edge weights
    plt.subplot(2, 1, 2)
    weights = nx.get_edge_attributes(G, "weight").values()
    plt.hist(weights,bins=20)
    plt.xlabel('Edge weight')
    plt.title("Histogram of Edge weights")

    plt.tight_layout()
    
    
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x
    


def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    """ 2 Layer Graph Convolutional Network.

    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GCN
    lr : float
        learning rate for GCN
    weight_decay : float
        weight decay coefficient (l2 normalization) for GCN.
        When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GCN will be linearized.
    with_bias: bool
        whether to include bias term in GCN weights.
    device: str
        'cpu' or 'cuda'.

    Examples
    --------
	We can first load dataset and then train GCN.

    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    >>> gcn = gcn.to('cpu')
    >>> gcn.fit(features, adj, labels, idx_train) # train without earlystopping
    >>> gcn.fit(features, adj, labels, idx_train, idx_val, patience=30) # train with earlystopping
    >>> gcn.test(idx_test)
    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, device=None, variance = None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
        self.dropout = dropout
 
        self.lr = lr
        self.test_ = False
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        if variance is not None:
            self.variance = variance.repeat(nhid, 1).T
            self.variance_two = variance.repeat(nclass, 1).T
        else: 
            self.variance = variance
            self.variance_two = variance

    def forward(self, x, adj):
        
            
        if self.with_relu:
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)
        

        if self.variance is not None and self.test_ == False: 
            x = x + torch.normal(torch.zeros_like(x), self.variance)

        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc2(x, adj)
       
        if self.variance is not None and self.test_ == False: 
            x = x + torch.normal(torch.zeros_like(x), self.variance_two)
        
            
        return F.log_softmax(x, dim=1)

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=500, **kwargs):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.

        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        normalize : bool
            whether to normalize the input adjacency matrix.
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """

        self.device = self.gc1.weight.device
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model W ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('=== training gcn model W===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            # def eval_class(output, labels):
            #     preds = output.max(1)[1].type_as(labels)
            #     return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro') + \
            #         f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

            # perf_sum = eval_class(output[idx_val], labels[idx_val])
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, idx_test):
        """Evaluate GCN performance on test set.

        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency

        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.


        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)
        
        
        
        
        
def get_graph(graph_name, attack_type = 'nettack', pre_attack_dataset = None, data_dir = '/home/shafi.z/GraphDiffusion/data/', ensure_connected = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if attack_type == 'grad':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open(data_dir + graph_name + '.pkl', 'rb') as file: 
            graph_dict = pkl.load(file)
       
        graph_dict['features'] = graph_dict['features'].to(device)
        graph_dict['labels'] = graph_dict['labels'].to(device)
        graph_dict['adj'] = graph_dict['adj'].to(device)
        graph = graph_dict['graph']
        if ensure_connected == True:
            if not nx.is_connected(graph):
            
                print ("Graph Disconnected")
                print ("Adding Random Edges to Connect Graph")
                
                cc = list(nx.connected_components(graph))
                cc.reverse()
                for i in range(len(cc) - 1):
        
                    node_one = np.random.choice(list(cc[i]))
                    node_two = np.random.choice(list(cc[-1]))
        
                    graph.add_edge(node_one, node_two)
                    print ("Adding Random Edge : ", node_one, node_two)
        nx.set_edge_attributes(graph, name = 'weight', values = {e : 1 for e in graph.edges})

        graph_dict['graph'] = graph

        return graph_dict
    
    if pre_attack_dataset is None:
        # 'cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'
        data = Dataset(root = '/tmp/', name = graph_name, setting = attack_type)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        idx_unlabeled = np.union1d(idx_val, idx_test)
    
    else: 
        print ("Using Attacked Data")
        labels, features, idx_train, idx_val, idx_test, adj = load_data(name = graph_name,
                                                                        dataset = pre_attack_dataset,
                                                                        attack = True, 
                                                                        data_dir = data_dir)
        idx_unlabeled = np.union1d(idx_val, idx_test)

    

    graph = nx.Graph(adj.todense())
    if ensure_connected == True:
        if not nx.is_connected(graph):
            
            print ("Graph Disconnected")
            print ("Adding Random Edges to Connect Graph")
            
            cc = list(nx.connected_components(graph))
            cc.reverse()
            for i in range(len(cc) - 1):
    
                node_one = np.random.choice(list(cc[i]))
                node_two = np.random.choice(list(cc[-1]))
    
                graph.add_edge(node_one, node_two)
                print ("Adding Random Edge : ", node_one, node_two)
            
    nx.set_edge_attributes(graph, name = 'weight', values = {e : 1 for e in graph.edges})
    
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()

    return_dict = {}
    return_dict['adj'] = torch.Tensor(adj.todense()).to(device)
    return_dict['features'] = features
    return_dict['labels'] = labels
    return_dict['idx_train'] = idx_train    
    return_dict['idx_val'] = idx_val
    return_dict['idx_test'] = idx_test
    return_dict['idx_unlabeled'] = idx_unlabeled
    return_dict['edge_index'] = edge_index
    return_dict['graph'] = graph
    
    return return_dict
def get_curvature_info(graph, ricci_alpha = 0.5, ricci_proc = 8, ricci_iter = 50):

    print ("Calculating Hop Distance", end = '\r')
    # Hop Distance 
    len_adj = []
    for n in tqdm(graph.nodes):
        lengths = dict(nx.single_source_dijkstra_path_length(graph, n))
        len_adj.append(np.array(list(dict(sorted(lengths.items(), key = lambda item: item[0])).values())))
    len_adj = np.array(len_adj)

    # print ("Calculating Curvature", end = '\r')
    # # Curvature
    # orc = OllivierRicci(graph, alpha=0.5, verbose = "TRACE")
    # orc.compute_ricci_curvature()
    # G_orc = orc.G.copy()

    print ("Calculating Flow", end = '\r')
    # Flow
    orf = OllivierRicci(graph, alpha = ricci_alpha, base = 1, exp_power = 0, proc = ricci_proc, verbose = "INFO")
    orf.compute_ricci_flow(iterations = ricci_iter)
    G_rf = orf.G.copy()

    # Flow based similarity
    print ("Calculating Flow Similarity", end = '\r')
    distances = nx.all_pairs_dijkstra_path_length(G_rf, weight = 'weight')
    dist_mat = np.zeros((len(G_rf), len(G_rf)))

    for i in tqdm(range(len(G_rf))):
        
        dist_dict = next(distances)
        
        for node in dist_dict[1]:
            dist_mat[dist_dict[0]][node] = dist_dict[1][node]

    return_dict = {}
    return_dict['dist_mat'] = dist_mat
    return_dict['len_adj'] = len_adj
    # return_dict['G_orc'] = G_orc
    return_dict['G_rf'] = G_rf

    return return_dict
    
            
def sample_graphs(graph, dist_mat, len_adj, sample_dict, debug = False):

    sigma = sample_dict['sigma']
    k = sample_dict['k']
    sample_count = sample_dict['sample_count']
    threshold = sample_dict['threshold']
    
    # Sample Train Graphs
    prob_mat = np.exp(- np.square(dist_mat) / (2 * np.square(sigma)))

    sampled_mat = np.zeros(prob_mat.shape)
    for _ in tqdm(range(sample_count)):

        sample = np.zeros(prob_mat.shape)
        for i in range(prob_mat.shape[0]):
            for j in range(prob_mat.shape[1]):
                sample[i][j] = probabilistic_round(prob_mat[i][j])
        np.fill_diagonal(sample, 0)
        
        # Ensure symmetry
        sample = sample + sample.T
        sample[sample > 1] = 1
        
        sampled_mat += sample
        
    # Filter out multiple samples
    sampled_mat[sampled_mat < threshold] = 0

    # Set everything else to 1
    sampled_mat[sampled_mat > 0] = 1

    # Only consider edges within k-hops of each other 
    sampled_mat[len_adj > k] = 0

    # Ensure symmetry
    sampled_mat = sampled_mat + sampled_mat.T
    sampled_mat[sampled_mat > 1] = 1

    if debug:
        
        print ("Number of edges in sampled graph : ", np.sum(sampled_mat) / 2)
        print ("Number of edges in original graph : ", np.sum(nx.to_numpy_array(graph)) / 2)
        print ("Number of differences : ", np.sum(np.abs(sampled_mat - nx.to_numpy_array(graph))) / 2)

        f, axs = plt.subplots(1, 2)
        axs[0].imshow(sampled_mat)
        axs[0].set_title('Sampled Graph')

        axs[1].imshow(nx.to_numpy_array(graph))
        axs[1].set_title('Original Graph')

        f.show()

    graph = nx.Graph(sampled_mat)
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    
    return graph, edge_index, sampled_mat

def get_variances(graph, dist_mat, sample_dict, sample_count = 100, eigen_step = 100, plot = False, types = ['eigen', 'ricci'], graph_dict = None, alpha = 0.05):

    if sample_dict is not None:
        sigma = sample_dict['sigma']
        k = sample_dict['k']
        threshold = sample_dict['threshold']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return_dict = {}

    if 'conformal' in types:
        return_dict['conformal_radii'] = get_conformal_radii(graph_dict, alpha)

    if 'eigen' in types:
        # Eigen Variances
        vals, vecs = np.linalg.eig((nx.to_numpy_array(graph)))
        recon_graphs = []
    
        for components in tqdm(range(10, len(graph), eigen_step)):
            recon_graphs.append(vecs[:, :components] @ np.diag(vals[:components]) @ vecs[:, :components].T)
    
        recon_graphs = np.array(recon_graphs)
        eig_variance = np.std(recon_graphs, axis = 0)
        eig_variance = 1 / np.mean(eig_variance, axis = 0)
        eig_variance = (eig_variance - np.min(eig_variance)) / np.ptp(eig_variance)
        eig_variance += 1e-4

        recon_adj = []
        for i in recon_graphs:
            recon_adj.append((i.real - np.min(i.real)) / np.ptp(i.real))
    
        a = np.mean(recon_adj, axis = 0).real
        cus_eig = 1 - np.mean(np.abs(a - (1 - a)), axis = 1)
        cus_eig = (cus_eig - np.min(cus_eig)) / np.ptp(cus_eig)
        cus_eig += 1e-4

        return_dict['binary_deviation'] = torch.tensor(cus_eig, dtype = torch.float32).to(device)
        return_dict['eig_variance'] = torch.tensor(eig_variance, dtype = torch.float32).to(device)
        return_dict['a'] = a
    if 'ricci' in types:
        
        # Sampled Graphs
        prob_mat = np.exp(- np.square(dist_mat) / (2 * np.square(sigma)))
    
        sampled_mat = np.zeros(prob_mat.shape)
        for _ in tqdm(range(sample_count)):
    
            sample = np.zeros(prob_mat.shape)
            for i in range(prob_mat.shape[0]):
                for j in range(prob_mat.shape[1]):
                    sample[i][j] = probabilistic_round(prob_mat[i][j])
            np.fill_diagonal(sample, 0)
            
            # Ensure symmetry
            sample = sample + sample.T
            sample[sample > 1] = 1
            
            sampled_mat += sample
    
        # Custom Variances
        sampled_mat = sampled_mat / sample_count
        variances = np.mean(np.abs(sampled_mat - (1 - sampled_mat)), axis = 1)
        variances = (variances - np.min(variances)) / np.ptp(variances)
        variances += 1e-4
    
        custom_variance = variances.copy()
    
        # Entropy Variances
        variances = entropy(sampled_mat, axis = 1)
        variances = np.nan_to_num(variances, 0)
        variances = (variances - np.min(variances)) / np.ptp(variances)
        variances = 1 - variances
    
        entropy_variance = variances.copy()
    
        # Standard Deviation Variances
        variances = 1 - np.std(sampled_mat, axis = 1)
        variances = (variances - np.min(variances)) / np.ptp(variances)
        variances += 1e-4
    
        std_variance = variances.copy()

        return_dict['custom_variance'] = torch.tensor(custom_variance, dtype = torch.float32).to(device)
        return_dict['entropy_variance'] = torch.tensor(entropy_variance, dtype = torch.float32).to(device)
        return_dict['std_variance'] = torch.tensor(std_variance, dtype = torch.float32).to(device)
        return_dict['sampled_mat'] = sampled_mat

    

    if plot: 
       
        if 'ricci' in types:
            plt.scatter(list(dict(graph.degree).values()), std_variance, s = 1, label = 'STD')
            plt.scatter(list(dict(graph.degree).values()), custom_variance, s = 1, label = '|A - (1 - A)|')
            plt.scatter(list(dict(graph.degree).values()), entropy_variance, s = 1, label = 'Entropy')
        if 'eigen' in types:
            # plt.scatter(list(dict(graph.degree).values()), eig_variance, s = 1, label = 'Eigen')
            plt.scatter(list(dict(graph.degree).values()), cus_eig, s = 1, label = 'Eigen - Binary Deviation')
        if 'conformal' in types: 
            conf =  return_dict['conformal_radii'].detach().cpu().numpy()
            # conf = (conf - np.min(conf)) / np.ptp(conf)
            plt.scatter(list(dict(graph.degree).values()), conf, s = 1, label = 'Conformal Radii')
        

        plt.legend()
        plt.ylabel('Variance')
        plt.xlabel('Degree')
        plt.xscale('log')
        # plt.yscale('log')
        plt.show()

    return return_dict
    
def attack_graph(graph_name, attack_setting, nhid = 16, custom_model = None):

    data = Dataset(root='/tmp/', name = graph_name, setting = attack_setting)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if custom_model is None: 

        surrogate = GCN(nfeat = features.shape[1],
                        nclass = labels.max().item() + 1,
                        nhid = nhid,
                        with_relu = False,
                        device = device)
        surrogate = surrogate.to(device)
        surrogate.fit(features, adj, labels, idx_train)
    
    else: 
        surrogate = custom_model

    attack_model = Metattack(model = surrogate,
                             nnodes = adj.shape[0],
                             feature_shape = features.shape,
                             device = device)
    attack_model = attack_model.to(device)

    perturbations = int(0.05 * (adj.sum() // 2))

    attack_model.attack(features, adj, labels, idx_train, idx_test, perturbations, ll_constraint=False)
    modified_adj = attack_model.modified_adj

    modified_adj = scipy.sparse.csr_matrix(modified_adj.detach().cpu().numpy())

    return modified_adj
    
    
def attack_rgcn(graph_name, attack_setting = 'prognn', attack_method = 'meta', nhid = 16, train_iters = 200, graph_dict = None):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
           
        if graph_dict is None: 
            data = Dataset(root='/tmp/', name = graph_name, setting = attack_setting)
            adj, features, labels = data.adj, data.features, data.labels
            idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

            print('==================')
            print('=== load graph perturbed by Zugner metattack (under prognn splits) ===')

            perturbed_data = PrePtbDataset(root='/tmp/',
                                            name = graph_name,
                                            attack_method = attack_method,
                                            ptb_rate = 0.05)
            perturbed_adj = perturbed_data.adj
            
        else: 
            features = scipy.sparse.csr_array(graph_dict['features'].cpu().numpy())
            labels = graph_dict['labels'].cpu().numpy()
            perturbed_adj = scipy.sparse.csr_array(graph_dict['adj'].cpu().numpy())
            idx_train = graph_dict['idx_train']
            idx_test = graph_dict['idx_test']
            idx_val = graph_dict['idx_val']
            

        # Setup RGCN Model
        model = RGCN(nnodes = perturbed_adj.shape[0],
                        nfeat = features.shape[1],
                        nclass = labels.max() + 1,
                        nhid = nhid,
                        device = device)

        model = model.to(device)

        model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters = train_iters, verbose = True)

        return perturbed_adj, model, model.test(idx_test)
    
def attack_simpgcn(graph_name, attack_setting = 'prognn', attack_method = 'meta', nhid = 16, train_iters = 200, graph_dict = None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    if graph_dict is None:
        data = Dataset(root='/tmp/', name = graph_name, setting = attack_setting)
        adj, features, labels = data.adj, data.features, data.labels
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

        print('==================')
        print('=== load graph perturbed by Zugner metattack (under prognn splits) ===')

        perturbed_data = PrePtbDataset(root='/tmp/',
                                        name = graph_name,
                                        attack_method = attack_method,
                                        ptb_rate = 0.05)
        perturbed_adj = perturbed_data.adj
    else: 
        features = scipy.sparse.csr_array(graph_dict['features'].cpu().numpy())
        labels = graph_dict['labels']
        perturbed_adj = graph_dict['adj']
        idx_train = graph_dict['idx_train']
        idx_test = graph_dict['idx_test']
        idx_val = graph_dict['idx_val']
        

    # Setup Defense Model
    model = SimPGCN(nnodes=features.shape[0], nfeat=features.shape[1], nhid=nhid, nclass=labels.max()+1, device=device)
    model = model.to(device)
    

    # using validation to pick model
    model.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=200, verbose=True)

    # You can use the inner function of model to test
    model.test(idx_test)

    return perturbed_adj, model


def train_variance_model(graph_dict, variance, variance_hyp = 1, nhid = 16, lr = 1e-3, train_iters = 2000, adjacency = None, patience = 250):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if adjacency is None: 
        adjacency = graph_dict['adj'] 
 
    features = graph_dict['features'] 
    labels = graph_dict['labels'] 
    idx_train = graph_dict['idx_train']     
    idx_val = graph_dict['idx_val'] 
    idx_test = graph_dict['idx_test'] 

    if variance is not None: 
        variance = variance * variance_hyp

    model = GCN(nfeat = features.shape[1],
                nclass = labels.max().item() + 1,
                nhid = nhid,
                with_relu = False,
                device = device,
                variance = variance,
                lr = lr)

    model = model.to(device)
    model.fit(features = features,
                adj = adjacency,
                labels = labels,
                idx_train = idx_train,
                idx_val = idx_val,
                verbose = True, 
                train_iters = train_iters, 
                patience = patience)

    model.test_ = True

    output = model.predict(features = features, 
                           adj = adjacency)

    acc_test = accuracy(output[idx_test], labels[idx_test])

    print ("Test Accuracy : ", acc_test)

    return model, acc_test


def train_sampled_variance_model(graph_dict, sample_dict, curvature_dict, variance, variance_hyp = 1, nhid = 16, lr = 1e-3, num_samples = 20, train_iters = 100, adjacency = None):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if adjacency is None: 
        adjacency = graph_dict['adj'] 
    else: 
        curvature_dict = get_curvature_info(graph = nx.Graph(adjacency),
                                            ricci_alpha = 0.5,
                                            ricci_proc = 16,
                                            ricci_iter = 50)

    features = graph_dict['features'] 
    labels = graph_dict['labels'] 
    idx_train = graph_dict['idx_train']     
    idx_val = graph_dict['idx_val'] 
    idx_test = graph_dict['idx_test'] 

    if variance is not None: 
        variance = variance * variance_hyp

    model = GCN(nfeat = features.shape[1],
                nclass = labels.max().item() + 1,
                nhid = nhid,
                with_relu = False,
                device = device,
                variance = variance,
                lr = lr)

    model = model.to(device)
    best_test = 0

    for i in range(num_samples):
            
            sampled_graph, edge_index, sampled_mat = sample_graphs(graph = graph_dict['graph'],
                                                                    dist_mat = curvature_dict['dist_mat'],
                                                                    len_adj = curvature_dict['len_adj'],
                                                                    sample_dict = sample_dict,
                                                                    debug = False)
            if i == 0: 
                init = True
            else: 
                init = False

            model.fit(features,
                        torch.Tensor(nx.to_numpy_array(sampled_graph)).to(device),
                        labels,
                        idx_train,
                        train_iters = train_iters,
                        initialize = init,
                        verbose = True)


            model.test_ = True
            output = model.predict(features = features, 
                                    adj = adjacency)
            acc_test = accuracy(output[idx_test], labels[idx_test])
            print ("Test Accuracy : ", acc_test.item())
            
            if acc_test.item() > best_test:
                best_test = acc_test.item()
                best_model = model
            model.test_ = False

    return best_model, best_test


def train_eigen_sampled_variance_model(graph_dict, sample_dict, curvature_dict, variance, variance_hyp = 1, nhid = 16, lr = 1e-4, eigen_count = 50, train_iters = 100, adjacency = None, patience = 10):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if adjacency is None: 
        adjacency = graph_dict['adj'] 

    features = graph_dict['features'] 
    labels = graph_dict['labels'] 
    idx_train = graph_dict['idx_train']     
    idx_val = graph_dict['idx_val'] 
    idx_test = graph_dict['idx_test'] 

    if variance is not None: 
        variance = variance * variance_hyp
        
    model = GCN(nfeat = features.shape[1],
                nclass = labels.max().item() + 1,
                nhid = nhid,
                with_relu = False,
                device = device,
                variance = variance,
                lr = lr)

    model = model.to(device)
    best_test = 0

    if type(adjacency) == torch.Tensor:
        vals, vecs = np.linalg.eig(adjacency.cpu().numpy())
    else:
        vals, vecs = np.linalg.eig(adjacency.todense())
        
    r = []
    early_stopping = patience
    for components in tqdm(range(5, vals.shape[0], eigen_count)):
            
            recon_graph = vecs[:, :components] @ np.diag(vals[:components]) @ vecs[:, :components].T
            recon_graph = np.array(recon_graph.real)
            sample = np.array(recon_graph > np.percentile(recon_graph, 99))

            
            if components == 5: 
                init = True
            else: 
                init = False

            model.fit(features,
                        torch.Tensor(sample).to(device),
                        #scipy.sparse.csr_matrix(sample),
                        labels,
                        idx_train,
                        train_iters = train_iters,
                        initialize = init,
                        verbose = True)


            model.test_ = True
            output = model.predict(features = features, 
                                    adj = adjacency)
            acc_test = accuracy(output[idx_test], labels[idx_test])
            acc_val = accuracy(output[idx_val], labels[idx_val])
        
            print ("Component : ", str(components), "Test Accuracy : ", acc_test.item())
            print ("Component : ", str(components), "Val Accuracy : ", acc_val.item())
        
            r.append([components, acc_test.item()])
        
            if acc_test.item() > best_test:
                best_test = acc_test.item()
                best_model = model
                patience = early_stopping
                best_comp = components
            else: 
                patience -= 1
                
            model.test_ = False

            if components > early_stopping and patience <= 0:
                print ("Early Stopping at", best_comp, "components, test accuracy :", best_test)
                break

    return best_model, best_test, r



# Step 3: Define the Quantile Loss Function
def quantile_loss(preds, target):

    ## Quanile Loss
    alpha = 0.05
    q1 = alpha / 2#0.05
    q2 = 0.5
    q3 = 1 - (alpha / 2)#0.95
    
    
    e1 = target - preds[0]
    e2 = target - preds[1]
    e3 = target - preds[2]
    
    loss_1 = torch.max((q1 - 1) * e1, q1 * e1)
    loss_2 = torch.max((q2 - 1) * e2, q2 * e2)
    loss_3 = torch.max((q3 - 1) * e3, q3 * e3)

    loss = loss_1 + loss_2 + loss_3 
    
    return torch.mean(loss)

class MLP(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, device=None):

        super(MLP, self).__init__()

        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = nhid
        self.nclass = nclass
        self.dropout = dropout
        self.lr = lr
        self.features = None

        self.lin1 = torch.nn.Linear(self.nfeat, self.hidden_sizes)
        self.lin2 = torch.nn.Linear(self.hidden_sizes, self.hidden_sizes)
        
        self.linout_q1 = torch.nn.Linear(self.hidden_sizes, self.nclass)
        self.linout_q2 = torch.nn.Linear(self.hidden_sizes, self.nclass)
        self.linout_q3 = torch.nn.Linear(self.hidden_sizes, self.nclass)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)

    def initialize(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.linout_q1.reset_parameters()
        self.linout_q2.reset_parameters()
        self.linout_q3.reset_parameters()
        
    
    def forward(self, x):
        
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p = self.dropout, training = self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, self.dropout, training = self.training)
        
        q1 = self.linout_q1(x)
        q2 = self.linout_q2(x)
        q3 = self.linout_q3(x)
        
        return q1, q2, q3

    def fit(self, x, y, idx_train, idx_val = None, labels = None, epochs = 5000, patience = 250):

        self.initialize()
        self.train()
        early = False

        if idx_val is not None:
            best_loss_val = 100
            best_acc_val = 0
        
        for e in range(epochs):

            self.optimizer.zero_grad()
            
            out = self.forward(x[idx_train])
            # loss = F.mse_loss(out, y)
            loss = quantile_loss(out, y[idx_train])
            loss.backward()
            print ("Epoch : ", e, " | Loss : ", loss.item(), end = '\r')
            early_stopping = patience
            
            if idx_val is not None:
                if e % 10 == 0:
                    self.eval()

                    val_out = F.log_softmax(self.forward(x)[1][idx_val], dim = 1)
                    acc_val = accuracy(val_out, labels[idx_val])
                    loss_val = quantile_loss(val_out, y[idx_val])
                    
                    if best_loss_val > loss_val:
                        best_loss_val = loss_val
                        weights = deepcopy(self.state_dict())
                        patience = early_stopping
                    else:
                        patience -= 1
        
                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        weights = deepcopy(self.state_dict())
                    
                    self.train()
                    
                if e > early_stopping and patience <= 0:
                    early = True
                    print ()
                    break
                
            self.optimizer.step()
        if early:
            print ("Early Stopping at Epoch", e) 
            self.load_state_dict(weights)


def get_conformal_radii(graph_dict, alpha = 0.05):

    
    features = graph_dict['features']
    labels = graph_dict['labels']
    adj = graph_dict['adj'].cpu()
    idx_train = graph_dict['idx_train']
    idx_test = graph_dict['idx_test']
    idx_val = graph_dict['idx_val']
    graph = graph_dict['graph']
    
    teacher = GCN(nfeat = features.shape[1],
                    nclass = labels.max().item() + 1,
                    nhid = 16,
                    with_relu = False,
                    device = device, 
                    lr = 5e-4)
    
    teacher = teacher.to(device)
    teacher.fit(features, adj, labels, idx_train, verbose = False, train_iters = 4000)
    teacher.return_prob = True
    targets = teacher.predict(features.to(device), adj.to(device)).detach().cpu().numpy()
    acc = teacher.test(idx_test)


    student = MLP(nfeat = features.shape[1],
                  nclass = labels.max().item() + 1,
                  nhid = 1024,
                  device = device, 
                  lr = 1e-4)
    student = student.to(device)
    targets = torch.Tensor(targets).to(device)

    student.fit(x = features,
                y = targets,
                idx_train = np.hstack([idx_train, idx_val, idx_test]),
                idx_val = idx_val,
                labels = labels,
                patience = 250)

    student.eval()
    student_out = student.forward(features)
    targets = targets.detach().cpu().numpy()
    test_acc = accuracy(F.log_softmax(student.forward(features[idx_test])[1], dim = 1), labels[idx_test]).item()
    print ("Teacher Acc : ", acc)
    print ("Student Acc : ", test_acc)


    g = adj.detach().cpu().numpy()
    vals, vecs = np.linalg.eig((g))
    
    components = 128
    recon = vecs[:, :components] @ np.diag(vals[:components]) @ vecs[:, :components].T
    recon = recon.real
    intersection = (recon > 0.5) * g
    conf_nodes = set(np.where(np.sum(intersection, axis = 1) > 1)[0])
    cal_set = np.array(list(set(idx_train).intersection(conf_nodes)), dtype = int)
    # cal_set = idx_train

    n = cal_set.shape[0]
    cal_labels = targets[cal_set]
    student_upper = student_out[2].detach().cpu().numpy()
    student_lower = student_out[0].detach().cpu().numpy()
    
    # Get scores
    cal_scores = np.maximum(cal_labels - student_upper[cal_set], student_lower[cal_set] - cal_labels)
    qhat = np.quantile(cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation = 'higher')
    
    student_lower = student_out[0].detach().cpu().numpy()
    student_upper = student_out[2].detach().cpu().numpy()
    
    prediction_sets = [student_lower - qhat, student_upper + qhat]
    prediction_sets = [student_lower, student_upper]
    
    conformal_radii = np.mean(prediction_sets[1] - prediction_sets[0], axis = 1)
    conformal_radii = (conformal_radii - np.min(conformal_radii)) / np.ptp(conformal_radii)

    return torch.tensor(conformal_radii, dtype = torch.float32).to(device)

def get_sense_features(graph, weighted = False, directed = False, is_bipartite = False):
    
    if is_bipartite:
        sense_feat_dict = {

            'Degree' : 0,
            'Average Neighbor Degree' : 1,
        }
    
    else: 
        sense_feat_dict = {

            'Degree' : 0,
            'Clustering Coefficient' : 1, 
            'Average Neighbor Degree' : 2,
            'Average Neighbor Clustering' : 3,
            'Node Betweenness' : 4,
            'Structural Holes Constraint' : 5,

        }
    
    ig = igraph.Graph([[e[0], e[1]] for e in nx.to_edgelist(graph)])
    sense_features = np.zeros((len(graph), len(sense_feat_dict)))

    if "Degree" in sense_feat_dict:
        print ("Calculating Degrees...                                   ", end = '\r')
        # Degree
        sense_features[:, sense_feat_dict['Degree']] = list(dict(graph.degree).values())

    if "Average Neighbor Degree" in sense_feat_dict:
        print ("Calculating Average Neighbor Degree...                    ", end = '\r')
        # Neighbor Degree Average
        sense_features[:, sense_feat_dict['Average Neighbor Degree']] = [np.mean([graph.degree[neighbor] for neighbor in dict(graph[node]).keys()]) for node in graph.nodes]

    if "Clustering Coefficient" in sense_feat_dict:
        print ("Calculating Clustering Coefficient...                     ", end = '\r')
        # Clustering Coefficient
        cluster_dict = nx.clustering(graph)
        sense_features[:, sense_feat_dict['Clustering Coefficient']] = list(cluster_dict.values())

    if "Average Neighbor Clustering" in sense_feat_dict:
        print ("Calculating Average Neighbor Clustering Coefficients...   ", end = '\r')
        # Neighbor Average Clustering 
        sense_features[:, sense_feat_dict['Average Neighbor Clustering']] = [np.mean([cluster_dict[neighbor] for neighbor in list(graph[node])]) for node in graph.nodes]
    
    if "Node Betweenness" in sense_feat_dict:
        print ("Calculating Node Betweenness...                           ", end = '\r')
        # Node Betweenness 
        sense_features[:, sense_feat_dict['Node Betweenness']] = ig.betweenness(directed = directed) #list(nx.algorithms.centrality.betweenness_centrality(graph).values())
    
    if "Structural Holes Constraint" in sense_feat_dict:
        print ("Calculating Structural Hole Constraint Scores...         ", end = '\r')
        # Structual Holes
        sense_features[:, sense_feat_dict['Structural Holes Constraint']] = ig.constraint() #list(nx.algorithms.structuralholes.constraint(graph, weight = 'weight').values())
     
    print ("Normalizing Features Between 0 And 1...                   ", end = '\r')
    # Normalise to between 0 and 1 
    sense_features = (sense_features - np.min(sense_features, axis = 0)) / np.ptp(sense_features, axis = 0)
    
    print ("Done                                                      ", end = '\r')
    
    sense_features[np.isnan(sense_features)] = 0
    return sense_feat_dict, sense_features
    
