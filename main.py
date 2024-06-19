from copy import deepcopy
import argparse
import math
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from loguru import logger
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sparsemax import Sparsemax
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset, random_split
from tsai.models.TCN import TemporalConvNet

from sklearn.metrics import precision_recall_curve, roc_auc_score, accuracy_score, balanced_accuracy_score, auc, roc_curve

logger.remove()
stderr_logger_id = logger.add(sys.stderr, level="INFO")

config_dict = {}
config_dict["run_name"] = "kdd_2024"
config_dict["subrun_name"] = "main"

save_path = Path("../runs", config_dict["run_name"])
save_path.mkdir(parents=True, exist_ok=True)
_model_save_path = "models"
(save_path / _model_save_path).mkdir(parents=True, exist_ok=True)
_graph_save_path = "graphs"
(save_path / _graph_save_path).mkdir(parents=True, exist_ok=True)
_result_save_path = "results"
(save_path / _result_save_path).mkdir(parents=True, exist_ok=True)

log_save_path = save_path / f'{config_dict["subrun_name"]}.log'
logger.add(log_save_path, level="TRACE")
logger.info(f"===== LOGGER =====")
logger.info(f"logger path: {log_save_path}")

config_dict["learning_rate"] = 0.001
config_dict["pred_len"] = 1
config_dict["eval_var_batch_size"] = 8
config_dict["reconstruction_epochs"] = 0
config_dict["joint_epochs"] = 2000
config_dict["stop_epoch"] = config_dict["reconstruction_epochs"] + config_dict["joint_epochs"]
config_dict["cuda_device"] = "cuda:0"
config_dict["lag"] = 1
config_dict["eval_frequency"] = 10
config_dict["print_frequency"] = 100
config_dict["eval_var_batch_size"] = 4

logger.info(f"all config yaml: {config_dict}")

def numpy2tensor(np_array):
    return torch.tensor(np_array).float()

# The Dynamic Causal Discovery Model

class VARP(nn.Module):
    def __init__(self, c_in, lag=1, encoder_layers=3*[6], decoder_layers=3*[6], kernel_size=3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.scale_factor = 1
        self.c_in = c_in
        self.lag = lag
        self.const_pad = nn.ConstantPad1d((lag - 1, 0), 0)
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder = TemporalConvNet(1, self.encoder_layers, kernel_size, dropout=0.2)
        self.decoder = TemporalConvNet(
            self.encoder_layers[-1] * 1,
            [c * 1 for c in self.decoder_layers],
            kernel_size,
            dropout=0.2
        )
        self.encoder_downsample = nn.AvgPool1d(self.scale_factor)
        self.var_mat = nn.parameter.Parameter(torch.zeros(self.c_in, self.c_in, self.encoder_layers[-1], lag).normal_(0, 0.01))
        self.var_relu = nn.ReLU()
        self.decoder_upsample = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
        self.decoder_1d_conv = nn.Conv1d(self.decoder_layers[-1] * 1, 1, 1)
    
    def forward(self, x, var_fusion_enabled):
        B, N, T = x.shape
        C = self.encoder_layers[-1]
        x = x.reshape(B * N, 1, T)                # B*N*T -> BN*1*T
        x = self.encoder(x)                       # BN*1*T -> BN*C*T
        
        if var_fusion_enabled:
            # VAR Fusion
            x = self.const_pad(x)
            x = x.unfold(2, self.lag, 1)
            x = x.reshape(
                B,
                N,
                C,
                -1,
                self.lag,
            )
            x = x.transpose(2, 3) # B*N*T*C*L

            # Channel-wise Relu Aggregation 
            x = torch.einsum('nmjkl,imkl->nijkm', x, self.var_mat)
            x = self.var_relu(x)
            x = x.sum(dim=-1)
            x = x.reshape( # BN*T*C
                B * N,
                -1,
                C,
            )
            x = x.transpose(1, 2) # BN*C*T
        
        x = self.decoder_upsample(x)
        x = self.decoder(x)                       # B*NC*T -> B*NC*T
        x = self.decoder_1d_conv(x)               # B*NC*T -> B*N*T
        x = x.reshape(                            # B*N*C*T -> B*NC*T
            B,
            N,
            -1,
        )
        return x    
    
    def get_trainable_params(self):
        params = [self.var_mat]
        return params

def compute_l1_loss(w):
    return torch.abs(w).mean()


def compute_l2_loss(w):
    return torch.square(w).mean()

# Metric Calculation

def prCal(scoreList, prk, rightOne):
    prkSum = 0
    for k in range(min(prk, len(scoreList))):
        if scoreList[k] in rightOne:
            prkSum = prkSum + 1
    denominator = min(len(rightOne), prk)
    return prkSum / denominator


def pr_stat(scoreList, rightOne, k=5):
    topk_list = range(1, k + 1)
    prkS = [0] * len(topk_list)
    for j, k in enumerate(topk_list):
        prkS[j] += prCal(scoreList, k, rightOne)
    return prkS


def print_prk_acc(prkS, acc):
    headers = ["PR@{}".format(i + 1) for i in range(10)] + ["MAP", "RS"]
    data = prkS[:10] + [np.mean(prkS)]
    data.append(acc)
    logger.info(tabulate([data], headers=headers, floatfmt="#06.4f"))

def print_proportional_prk_acc(prkS, acc, trace=True):
    ratios = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25]
    k_indices = [math.ceil(ratio * len(prkS)) for ratio in ratios]
    k_indices = [i for i in range(10)]
    headers = [f"{ratios[i] * 100:02.1f}%/{k_indices[i]:02d}" for i in range(len(k_indices))] + ["MAP", "RS"]
    data = extract_proportional_prk_acc(prkS, acc)
    if trace:
        logger.trace("\n" + tabulate([data], headers=headers, floatfmt="#06.4f"))
    else:
        logger.info("\n" + tabulate([data], headers=headers, floatfmt="#06.4f"))

def extract_proportional_prk_acc(prkS, acc):
    ratios = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25]
    k_indices = [math.ceil(ratio * len(prkS)) for ratio in ratios]
    k_indices = [i for i in range(10)]
    data = [prkS[k] for k in k_indices] + [np.mean(prkS)]
    data.append(acc)
    return data


def rankscore(node_rank, rightOne):
    """Accuracy for Root Cause Analysis with multiple causes.
    Refined from the Acc metric in TBAC paper.
    """
    n = len(node_rank)
    s = 0.0
    for i in range(len(rightOne)):
        if rightOne[i] in node_rank:
            rank = node_rank.index(rightOne[i]) + 1
            s += (n - max(0, rank - len(rightOne))) / n
        else:
            s += 0
    s /= len(rightOne)
    return s

def eval_result(node_rank, ground_truth, k):
    prk_list = pr_stat(node_rank, ground_truth, k)
    rank_score = rankscore(node_rank, ground_truth)
    # print_prk_acc(prk_list, rank_score)
    return prk_list, rank_score


def sparsify_tensor(tensor, top_n=100):
    shape = tensor.shape
    tensor = tensor.flatten()
    values, indices = tensor.sort(descending=True)
    tensor[indices[top_n:]] = 0
    tensor = tensor.reshape(shape)
    return tensor

# ===== MAIN LOGIC LOOP =====
# logger.disable("__main__")
def load_varp_net(model_save_filename, config_dict, seed=1):
    # ===== Init New Model =====
    device = torch.device(config_dict["cuda_device"])
    lag = config_dict["lag"]
    net = VARP(nvars, lag, seed=seed).to(device)
    # ===== Load Model =====
    net.load_state_dict(torch.load(model_save_filename))
    return net

def train_varp_net(data_loader, config_dict, seed=1):
    # ===== Init New Model =====
    device = torch.device(config_dict["cuda_device"])
    lag = config_dict["lag"]
    net = VARP(nvars, lag, seed=seed).to(device)

    # ===== Training Model =====
    criterion = nn.MSELoss()
    mse_sum = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(net.parameters(), lr=config_dict["learning_rate"])

    epochs = config_dict["stop_epoch"]
    early_stop_counter = 0
    reconstruction_train_loss_list, reconstruction_val_loss_list = [], []
    joint_train_loss_list, joint_val_loss_list = [], []
    loss_min = float("inf")

    logger.info(f'RECONSTRUCTION STAGE')
    for epoch in range(epochs):  # loop over the dataset multiple times
        # --- RECONSTRUCTION TRAINING ---
        if epoch < config_dict["reconstruction_epochs"]:
            train_loss_epoch = [0.0] * 3
            net.train()
            net.encoder.train()
            net.decoder.train()
            for i, (X, Y) in enumerate(data_loader):
                # get the inputs; data is a array of [batch_size, channel, length]
                X, Y = X.to(device), Y.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                loss = 0
                # forward + backward + optimize

                # reconstruction loss
                reconstructed_output = net(X, False)
                reconstruction_loss = criterion(reconstructed_output, X)
                loss += reconstruction_loss

                # l1 regularization
                l1_weight = 1
                l1_loss = 0
                l1_n = 0
                for param in net.get_regularized_params(stage=0):
                    l1_loss += compute_l1_loss(param)
                    l1_n += param.numel()
                l1_loss = l1_loss * l1_weight
                loss += l1_loss

                loss.backward()
                optimizer.step()

                # print statistics
                train_loss_epoch[1] += reconstruction_loss.item()
                train_loss_epoch[2] += l1_loss.item()
                

            reconstruction_train_loss_list.append([loss / (i + 1) for loss in train_loss_epoch])

            # --- EVALLING ---
            val_loss_epoch = 0.0
            net.eval()
            net.encoder.eval()
            net.decoder.eval()
            eval_n = 0
            with torch.inference_mode():
                for j, (X, Y) in enumerate(data_loader):
                    # get the inputs; data is a array of [batch_size, channel, length]
                    X, Y = X.to(device), Y.to(device)

                    # forward only
                    reconstructed_output = net(X, False)
                    reconstruction_loss = mse_sum(reconstructed_output, X)
                    # print statistics
                    val_loss_epoch += reconstruction_loss.item()
                    eval_n += X.numel()
            
            val_loss_epoch = val_loss_epoch / eval_n
            reconstruction_val_loss_list.append(val_loss_epoch)
            # early stopping
            if loss_min > val_loss_epoch:
                loss_min = val_loss_epoch
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            if epoch == 0 or epoch % config_dict["print_frequency"] == 0 or epoch == config_dict["reconstruction_epochs"] - 1:
                logger.info(f'[{epoch + 1:4d}] train: {sum(train_loss_epoch) / (i + 1):.5f} {train_loss_epoch[0] / (i + 1):.5f} {train_loss_epoch[1] / (i + 1):.5f} {train_loss_epoch[2] / (i + 1):.5f} | val: {val_loss_epoch / (j + 1):.6f} | {early_stop_counter:2}')

        # --- JOINT TRAINING ---
        if epoch == config_dict["reconstruction_epochs"]:
            logger.info(f'JOINT STAGE')
            loss_min = float("inf")
        if epoch >= config_dict["reconstruction_epochs"]:
            train_loss_epoch = [0.0] * 3
            net.train()
            net.encoder.train()
            net.decoder.train()
            for i, (X, Y) in enumerate(data_loader):
                # get the inputs; data is a array of [batch_size, channel, length]
                X, Y = X.to(device), Y.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                loss = 0
                # forward + backward + optimize

                # prediction loss
                predicted_output = net(X, True)
                prediction_loss = criterion(predicted_output, Y)
                loss += prediction_loss

                # reconstruction loss
                reconstructed_output = net(X, False)
                reconstruction_loss = criterion(reconstructed_output, X)
                loss += reconstruction_loss

                # l1 regularization
                l1_weight = 1
                l1_loss = 0
                l1_n = 0
                for param in net.get_regularized_params():
                    l1_loss += compute_l1_loss(param)
                    l1_n += param.numel()
                # l1_loss = l1_loss / l1_n * l1_weight
                l1_loss = l1_loss * l1_weight
                loss += l1_loss

                loss.backward()
                optimizer.step()

                # print statistics
                train_loss_epoch[0] += prediction_loss.item()
                train_loss_epoch[1] += reconstruction_loss.item()
                train_loss_epoch[2] += l1_loss.item()
                

            joint_train_loss_list.append([loss / (i + 1) for loss in train_loss_epoch])

            # --- EVALLING ---
            val_loss_epoch = 0.0
            net.eval()
            net.encoder.eval()
            net.decoder.eval()
            eval_n = 0
            with torch.inference_mode():
                for j, (X, Y) in enumerate(data_loader):
                    # get the inputs; data is a array of [batch_size, channel, length]
                    X, Y = X.to(device), Y.to(device)

                    # forward only
                    predicted_output = net(X, True)
                    prediction_loss = mse_sum(predicted_output, Y)
                    # print statistics
                    val_loss_epoch += prediction_loss.item()
                    eval_n += X.numel()
            
            val_loss_epoch = val_loss_epoch / eval_n
            joint_val_loss_list.append(val_loss_epoch)
            
            # early stopping
            if loss_min > val_loss_epoch:
                loss_min = val_loss_epoch
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            # if True:
            if epoch == config_dict["reconstruction_epochs"] or epoch % config_dict["print_frequency"] == 0 or epoch == config_dict["stop_epoch"] - 1:
                logger.info(f'[{epoch + 1:4d}] | {early_stop_counter:2} | trainL: {sum(train_loss_epoch) / (i + 1):.5f} {train_loss_epoch[0] / (i + 1):.5f} {train_loss_epoch[1] / (i + 1):.5f} {train_loss_epoch[2] / (i + 1):.5f} | valL: {val_loss_epoch / (j + 1):.6f}')

    return net, reconstruction_val_loss_list, joint_val_loss_list, reconstruction_train_loss_list, joint_train_loss_list

from collections import defaultdict
import yaml
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from scipy.stats import geom
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import math
import warnings
import networkx as nx
warnings.filterwarnings('ignore', category=RuntimeWarning)
from loguru import logger

## ==== RCA Lib ====
import time
from scipy.stats import geom

def eval_rca_results(node_rank, ground_truth, N):
    prk_list, rank_score = eval_result(node_rank, ground_truth, N)
    return prk_list, rank_score

def extract_causal_graph(net, df, data_loader, config_dict, stochastic_pertubations=128):
    # print("----a----")
    # start_time = time.time()
    net.eval()
    config_dict["eval_var_batch_size"] = 100
    device = torch.device(config_dict["cuda_device"])
    import math

    nvars = len(df.columns)
    full_error_expanded, red_error_expanded = torch.zeros(nvars, nvars, len(df) - config_dict["pred_len"]).to(device), torch.zeros(nvars, nvars, len(df) - config_dict["pred_len"]).to(device)
    for batch_ind, (X, Y_truth) in enumerate(data_loader):
        X, Y_truth = X.to(device), Y_truth.to(device)
        for _ in range(stochastic_pertubations):
            for j in range(len(X)):
                X_ = X[j].repeat(nvars, 1, 1)
                for i in range(nvars):
                    X_[i, i, :] = X_[i, i, torch.randperm(X_.shape[2])]
                with torch.inference_mode():
                    Y = net(X[j].unsqueeze(0), True)
                    Y_ = torch.zeros(nvars, nvars, len(df) - config_dict["pred_len"]).to(device)
                    for eval_batch in range(math.ceil(nvars / config_dict["eval_var_batch_size"])):
                        # print(batch_ind, j, eval_batch)
                        Y_[config_dict["eval_var_batch_size"] * eval_batch : 
                            config_dict["eval_var_batch_size"] * (eval_batch + 1)] = net(X_[config_dict["eval_var_batch_size"] * eval_batch : 
                                                                                                config_dict["eval_var_batch_size"] * (eval_batch + 1)].to(device), True)
                full_error_expanded += torch.square(Y[:, :] - Y_truth[j, :, :])
                red_error_expanded += torch.square(Y_[:, :, :] - Y_truth[j, :, :].repeat(len(df.columns), 1, 1))
    error_difference_expanded = (red_error_expanded - full_error_expanded) / stochastic_pertubations
    error_difference_expanded[error_difference_expanded < 0] = 0
    return error_difference_expanded.cpu()

def downsample_causal_graph(causal_graph, pool_segments):
    # causal_graph: M*M*T
    pool_segments = min(causal_graph.shape[2], pool_segments)
    pool_stride = causal_graph.shape[2] / pool_segments
    causal_graph_segmented = np.zeros((nvars, nvars, pool_segments))

    for i in range(pool_segments):
        range_start, range_end = pool_stride * i, pool_stride * (i + 1) - 1
        range_start_floor, range_end_ceil = math.floor(range_start), math.ceil(range_end)
        filter_window = np.zeros((causal_graph.shape[2],))
        filter_window[range_start_floor:range_end_ceil + 1] = 1 / pool_stride
        filter_window[range_start_floor] *= range_start_floor + 1 - range_start
        filter_window[range_end_ceil] *= range_end + 1 - range_end_ceil
        causal_graph_segmented[:, :, i] = (causal_graph[:, :, :] * filter_window[None, None, :]).sum(axis=-1)

    return causal_graph_segmented

def pagerank_scipy_speedup(
    graph,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):
    """Returns the PageRank of the nodes in the graph.

    PageRank computes a ranking of the nodes in the graph G based on
    the structure of the incoming links. It was originally designed as
    an algorithm to rank web pages.

    Parameters
    ----------
    G : graph
      A NetworkX graph.  Undirected graphs will be converted to a directed
      graph with two directed edges for each undirected edge.

    alpha : float, optional
      Damping parameter for PageRank, default=0.85.

    personalization: dict, optional
      The "personalization vector" consisting of a dictionary with a
      key some subset of graph nodes and personalization value each of those.
      At least one personalization value must be non-zero.
      If not specified, a nodes personalization value will be zero.
      By default, a uniform distribution is used.

    max_iter : integer, optional
      Maximum number of iterations in power method eigenvalue solver.

    tol : float, optional
      Error tolerance used to check convergence in power method solver.

    nstart : dictionary, optional
      Starting value of PageRank iteration for each node.

    weight : key, optional
      Edge data key to use as weight.  If None weights are set to 1.

    dangling: dict, optional
      The outedges to be assigned to any "dangling" nodes, i.e., nodes without
      any outedges. The dict key is the node the outedge points to and the dict
      value is the weight of that outedge. By default, dangling nodes are given
      outedges according to the personalization vector (uniform if not
      specified) This must be selected to result in an irreducible transition
      matrix (see notes under google_matrix). It may be common to have the
      dangling dict to be the same as the personalization dict.

    Returns
    -------
    pagerank : dictionary
       Dictionary of nodes with PageRank as value

    Examples
    --------
    >>> from networkx.algorithms.link_analysis.pagerank_alg import _pagerank_scipy
    >>> G = nx.DiGraph(nx.path_graph(4))
    >>> pr = _pagerank_scipy(G, alpha=0.9)

    Notes
    -----
    The eigenvector calculation uses power iteration with a SciPy
    sparse matrix representation.

    This implementation works with Multi(Di)Graphs. For multigraphs the
    weight between two nodes is set to be the sum of all edge weights
    between those nodes.

    See Also
    --------
    pagerank

    Raises
    ------
    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Page, Lawrence; Brin, Sergey; Motwani, Rajeev and Winograd, Terry,
       The PageRank citation ranking: Bringing order to the Web. 1999
       http://dbpubs.stanford.edu:8090/pub/showDoc.Fulltext?lang=en&doc=1999-66&format=pdf
    """
    import numpy as np
    import scipy as sp
    import scipy.sparse  # call as sp.sparse

    N = graph.shape[0]
    if N == 0:
        return {}

    nodelist = list(range(graph.shape[0]))
    # start_time = time.time()
    A =  sp.sparse.coo_array((graph))
    # A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)
    # print(f"Converting --- {time.time() - start_time:.4f} seconds ---")

    # A = sp.sparse.coo_array((data, (row, col)), shape=(nlen, nlen), dtype=dtype)
    S = A.sum(axis=1)
    S[S != 0] = 1.0 / S[S != 0]
    # TODO: csr_array
    Q = sp.sparse.csr_array(sp.sparse.spdiags(S.T, 0, *A.shape))
    A = Q @ A

    # initial vector
    if nstart is None:
        x = np.repeat(1.0 / N, N)
    else:
        x = np.array([nstart.get(n, 0) for n in nodelist], dtype=float)
        x /= x.sum()

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N)
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()
    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = np.where(S == 0)[0]

    # power iteration: make up to max_iter iterations
    # start_time = time.time()
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x @ A + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
        # check convergence, l1 norm
        err = np.absolute(x - xlast).sum()
        if err < N * tol:
            # print(f"Solving --- {time.time() - start_time:.4f} seconds ---")
            return dict(zip(nodelist, map(float, x)))
    raise nx.PowerIterationFailedConvergence(max_iter)

def pagerank_rca(causal_graph_segmented, ground_truth, df, pool_segments=5, print_ranking=False):
    intermixed_graph = np.zeros((len(df.columns) * pool_segments, len(df.columns) * pool_segments))
    rv = geom(0.85)
    pmf = rv.pmf(np.arange(1, pool_segments + 1))
    pmf = pmf / pmf.sum()

    ground_truth = ground_truth_indices
    for present_t in range(pool_segments): # Present T
        for future_t in range(present_t, pool_segments): # Future T
            intermixed_graph[present_t * len(df.columns):(present_t + 1) * len(df.columns),
                            future_t * len(df.columns):(future_t + 1) * len(df.columns)] = causal_graph_segmented[:, :, future_t] * pmf[future_t - present_t]
            if future_t > present_t:
                for t_i in range(len(df.columns)):
                    intermixed_graph[present_t * len(df.columns) + t_i, future_t * len(df.columns) + t_i] = 0
    intermixed_graph = intermixed_graph.T
    personalization = {}
    for i in range(len(df.columns)):
        for j in range(pool_segments):
            # personalization[i + len(df.columns) * j] = j + 1
            personalization[i + len(df.columns) * j] = 1
            personalization[i + len(df.columns) * j] *= causal_graph_segmented[i, i, j]
    # personalization = None
    result_dict = pagerank_scipy_speedup(intermixed_graph)

    # Merged Version
    result_list = [(val, i) for i, val in result_dict.items()]
    result_list = sorted(result_list, reverse=True)
    node_rank = [k for _, k in result_list]

    agg_result_list =  [sum((result_dict[i + len(df.columns) * j] for j in range(pool_segments))) for i in range(len(df.columns))]
    agg_max_list = [np.argmax([result_dict[i + len(df.columns) * j] for j in range(pool_segments)]) for i in range(len(df.columns))]
    # agg_result_list = sorted(agg_result_list, reverse=True)
    # agg_node_rank = [k for _, k in agg_result_list]

    # print("*** separate results ***")
    # for v, k in result_list:
    #     print(f"{v:.4f} {k:3} {k // len(df.columns)} {df.columns[k % len(df.columns)]}")

    if print_ranking:
        logger.info("*** aggregated results ***")
        for v, k in agg_result_list:
            logger.info(f"{v:.4f} {k:3} {agg_max_list[k]} {df.columns[k]}")
    # prk_list, rank_score = eval_result(agg_node_rank, ground_truth, len(df.columns))
    return list(zip(agg_result_list, agg_max_list))

def calculate_proportional_prk(prk_list):
    ratios = [0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25]
    k_indices = [math.ceil(ratio * len(prk_list)) for ratio in ratios]
    proportional_prks = [prk_list[k_indices[i]] for i in range(len(k_indices))]
    return proportional_prks

## ==== Dataloader ====
class PredictionDataset(Dataset):
    def __init__(self, X, Y):
        super(PredictionDataset, self).__init__()
        self.X = X
        self.Y = Y
        assert X.shape == Y.shape

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def prepare_dataloader(df):
    seq_len = 512
    seq_stride = 512
    pred_len = 1
    batch_size = 8

    assert df.isnull().sum().sum() == 0
    if df.shape[0] >= seq_len:
        X_rolling = np.lib.stride_tricks.sliding_window_view(df.to_numpy()[:-pred_len, :], seq_len, axis=0)
        X = X_rolling[::seq_stride, :, :]

        Y_rolling = np.lib.stride_tricks.sliding_window_view(df.to_numpy()[pred_len:, :], seq_len, axis=0)
        Y = Y_rolling[::seq_stride, :, :]
    else:
        seq_len = len(df) - pred_len
        X = (df.iloc[:-pred_len, :].to_numpy()).T[None, ...]
        Y = (df.iloc[pred_len:, :].to_numpy()).T[None, ...]

    X = torch.from_numpy(X.copy()).float()
    Y = torch.from_numpy(Y.copy()).float()
    data_loader = DataLoader(PredictionDataset(X, Y), batch_size=batch_size, shuffle=True)
    return data_loader

## ==== Main ====
with open("../data_config.yaml", "r") as f:
    dataset_dict = yaml.safe_load(f)

case_rca_result = defaultdict(dict)
for i, case_name in enumerate(dataset_dict):
    logger.info(f"===== {case_name} =====")
    case_rca_result[case_name] = defaultdict(list)

    case_dict = dataset_dict[case_name]
    time_range_start_str = case_dict["time_range"]["start"]
    time_range_end_str = case_dict["time_range"]["end"]
    time_range_start = datetime.strptime(time_range_start_str, "%Y-%m-%d %H:%M")
    time_range_end = datetime.strptime(time_range_end_str, "%Y-%m-%d %H:%M")
    time_mid = time_range_start + (time_range_end - time_range_start) / 2
    expanded_start = time_range_start - timedelta(
        seconds=5 * 30
    )
    expanded_end = time_range_end + timedelta(
        seconds=5 * 30
    )
    logger.info(
        f"mid time ({time_mid.strftime('%Y-%m-%d %H:%M')})"
        f" start time ({expanded_start.strftime('%Y-%m-%d %H:%M')})"
        f" end time ({expanded_end.strftime('%Y-%m-%d %H:%M')})"
        f" length: {(expanded_end - expanded_start) / timedelta(minutes=1)} min(s)"
    )

    df = pd.read_parquet(f"../data/{case_name}.parquet")

    df = preprocess_df(df)

    ground_truth_indices = [df.columns.get_loc(m) for m in case_dict["root_causes"]]

    nvars = len(df.columns)
    data_loader = prepare_dataloader(df)

    pool_segments_list = [len(df) - config_dict["pred_len"]]
    seeds = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    # ++++++++ (1) Training Models ++++++++
    logger.info(f"++++++++ Training Models ++++++++")
    case_val_loss_list = []
    for seed in seeds:
        logger.info(f" >>> seed={seed} <<<")
        net, reconstruction_val_loss_list, joint_val_loss_list, reconstruction_train_loss_list, joint_train_loss_list = train_varp_net(data_loader, config_dict, seed=seed)
        case_val_loss_list.append(joint_val_loss_list)
        (save_path / _model_save_path / f"case{case_name}").mkdir(parents=True, exist_ok=True)
        model_save_filename = save_path / _model_save_path / f"case{case_name}" / f"seed{seed}.pt"
        torch.save(net.state_dict(), model_save_filename)
        logger.info(f"model saved to: {model_save_filename}")
    case_val_matrix = np.array(case_val_loss_list).mean(axis=0)
    np.save(save_path / _model_save_path / f"case{case_name}" / f"val_loss", case_val_matrix)
    
    # ++++++++ (2) Extracting Causal Graphs +++++++++
    # stochastic_pertubations = 128
    # pool_segments = len(df)
    # for seed in seeds:
    #     model_save_filename = save_path / _model_save_path / f"case{case_name}" / f"seed{seed}.pt"
    #     net = load_varp_net(model_save_filename, config_dict, seed)
    #     anomaly_causal_graph = extract_causal_graph(net, df, data_loader, config_dict, stochastic_pertubations, pool_segments)
    #     anomaly_causal_graph_static = anomaly_causal_graph.mean(axis=2)
    #     # normal_causal_graph = extract_causal_graph(net, normal_df, normal_data_loader, config_dict, stochastic_pertubations, 1)[:, :, 0]
    #     anomaly_graph_save_filename = save_path / _graph_save_path / f"case{case_name}" / f"anomaly_seed{seed}"
    #     # normal_graph_save_filename = save_path / _graph_save_path / f"case{case_name}" / f"normal_seed{seed}"
    #     np.save(anomaly_graph_save_filename, anomaly_causal_graph)
    #     # np.save(normal_graph_save_filename, normal_causal_graph)
    #     logger.info(f"Case {case_name} Seed {seed} Causal Graph Extracted.")


    # # ++++++++ (3) Dynamic Root Cause Analysis ++++++++
    # logger.info(f"++++++++ Dynamic Root Cause Analysis ++++++++")
    #     case_rca_result[case_name][seed] = {}
    #     for pool_segments in pool_segments_list:
    #         case_rca_result[case_name][seed][pool_segments] = pagerank_rca(anomaly_causal_graph, ground_truth=None, df=df, pool_segments=pool_segments, print_ranking=False)
    #         logger.info(f" ** Pools {pool_segments} done!")
    #     logger.info(f" -- Seed {seed} done!")
        
    logger.info(f">>> Case {case_name} done!")