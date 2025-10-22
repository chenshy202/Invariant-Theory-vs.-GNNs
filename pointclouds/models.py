from torch_geometric.nn import GINConv, GCNConv, global_mean_pool, global_max_pool, Set2Set, MessagePassing, PointGNNConv
from torch_geometric.nn.models import MLP
from torch.nn import Parameter
from torch.nn import ModuleList
from torch_geometric.utils import remove_self_loops, dropout_adj
from torch_scatter import scatter_mean
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torchmetrics.functional.pairwise import pairwise_euclidean_distance
import torch
import torch.nn as nn
from torch_scatter import scatter_add

class SiameseRegressor(torch.nn.Module):
    def __init__(self, ScalarModel):
        super(SiameseRegressor, self).__init__()
        self.SM = ScalarModel
        #metric learning (TODO: should we do this for the GW dist instead?)
        self.metric = nn.Linear(self.SM.out_dim, self.SM.out_dim, bias=False)
        self.metric.reset_parameters()
        self.linear = nn.Linear(1,1)
        self.linear.reset_parameters()

    def forward(self, x1, x2):
        n = x1.num_graphs
        emb1 = self.SM(x1) #n by out_dim
        emb2 = self.SM(x2) #n by out_dim
        emb1 = self.metric(emb1) # project
        emb2 = self.metric(emb2) #project
        dist_mat = pairwise_euclidean_distance(emb1, emb2) #n by n matrix
        dist_vec = dist_mat.flatten().unsqueeze(1)
        dist_pred = self.linear(dist_vec)
        return dist_pred

class PointGNN(nn.Module):
    def __init__(self, node_feature_dim, hid_dim, out_dim):
        super(PointGNN, self).__init__()

        self.out_dim = out_dim

        self.feature_encoder = nn.Sequential(
            nn.Linear(4, node_feature_dim)
        )

        self.mlp_h1 = nn.Sequential(
            nn.Linear(node_feature_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 3)
        )
        self.mlp_f1 = nn.Sequential(
            nn.Linear(node_feature_dim + 3, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )
        self.mlp_g1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hid_dim),
            nn.Linear(hid_dim, hid_dim)
        )
        self.conv1 = PointGNNConv(mlp_h=self.mlp_h1, mlp_f=self.mlp_f1, mlp_g=self.mlp_g1)


        self.mlp_h2 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 3)
        )
        self.mlp_f2 = nn.Sequential(
            nn.Linear(hid_dim + 3, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )
        self.mlp_g2 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hid_dim),
            nn.Linear(hid_dim, hid_dim)
        )
        self.conv2 = PointGNNConv(mlp_h=self.mlp_h2, mlp_f=self.mlp_f2, mlp_g=self.mlp_g2)

        # ------ readout ------
        self.mlp_readout = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, data):
        pos = data.pos  # [num_nodes, 3]
        batch = data.batch 
        edge_index = data.edge_index
        h = data.h

        h = self.feature_encoder(h)

        # PointGNN layers
        h = self.conv1(h, pos, edge_index)
        h = self.conv2(h, pos, edge_index)

        h = global_mean_pool(h, batch)


        h = self.mlp_readout(h)
        return h

    def reset_parameters(self):
        def reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

        for module in [
            self.feature_encoder,
            self.mlp_h1, self.mlp_f1, self.mlp_g1, self.conv1,
            self.mlp_h2, self.mlp_f2, self.mlp_g2, self.conv2,
            self.mlp_readout
        ]:
            module.apply(reset)
