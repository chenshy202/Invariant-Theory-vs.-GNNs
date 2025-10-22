import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, global_mean_pool, global_max_pool, Set2Set, MessagePassing
from torch_geometric.nn.models import MLP
from torch_geometric.nn import GINEConv
from torch.nn import Parameter
from torch.nn import ModuleList
from torch_geometric.utils import remove_self_loops, dropout_adj
from torch_scatter import scatter_mean, scatter_add

class GIN(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_dim, num_layers=2, max_nodes=23):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.node_feature_dim = node_feature_dim  
        self.out_dim = out_dim
        self.max_nodes = max_nodes
        

        for i in range(num_layers):
            mlp = MLP([node_feature_dim if i == 0 else hidden_dim, hidden_dim, hidden_dim], act="ReLU", batch_norm=True)
            self.convs.append(GINConv(nn=mlp))

        self.final_mlp = MLP([hidden_dim, hidden_dim, out_dim], act="ReLU", batch_norm=True)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.f_d, data.edge_index, data.edge_attr, data.batch

        for conv in self.convs:
            x = conv(x, edge_index) 
            x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.final_mlp(x)
        return x


class GINEConv_Weighted(MessagePassing):
    def __init__(self, 
                 nn_node,      
                 hidden_dim,   
                 eps=0.0, 
                 train_eps=False):
        super().__init__(aggr='add')  

        self.nn_node = nn_node
        self.train_eps = train_eps
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.eps = eps

        self.mlp_edge = nn.Sequential(
            nn.Linear(nn_node.in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nn_node.in_channels) 
        )

    def forward(self, x, edge_index, edge_attr):

        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.nn_node((1 + self.eps) * x + aggr_out)
        return out

    def message(self, x_j, edge_attr):

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)  

        z = x_j * edge_attr  
        m_ij = self.mlp_edge(z)  
        return m_ij

class GIN_weighted(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_dim, edge_dim=100, num_layers=2):

        super().__init__()
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.convs = ModuleList()

        for i in range(num_layers):
            in_dim = node_feature_dim if i == 0 else hidden_dim
            mlp_node = MLP([in_dim, hidden_dim, hidden_dim], act="ReLU", batch_norm=True)
            
            conv = GINEConv_Weighted(nn_node=mlp_node, 
                            hidden_dim=hidden_dim,
                            eps=0.0, 
                            train_eps=False)
                            
            self.convs.append(conv)

        self.final_mlp = MLP([hidden_dim, hidden_dim, out_dim], act="ReLU", batch_norm=True)

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.f_d, data.edge_index_A1, data.edge_attr1, data.batch
    

        # edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.final_mlp(x)
        return x


class GINEConv_concat(MessagePassing):

    def __init__(self, 
                 nn_node,     
                 edge_dim,     
                 hidden_dim,  
                 eps=0.0, 
                 train_eps=False):
        super().__init__(aggr='add') 

        self.nn_node = nn_node
        self.train_eps = train_eps
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.eps = eps

        self.mlp_edge = nn.Sequential(
            nn.Linear(nn_node.in_channels + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nn_node.in_channels)  
        )

    def forward(self, x, edge_index, edge_attr):

        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.nn_node((1 + self.eps) * x + aggr_out)
        return out

    def message(self, x_j, edge_attr):
        z = torch.cat([x_j, edge_attr], dim=-1)
        return self.mlp_edge(z)

class GIN_concat(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_dim, edge_dim=100, num_layers=2):
        super().__init__()

        # self.atom_encoder = nn.Linear(node_feature_dim, hidden_dim)
        # self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        self.num_layers = num_layers
        self.out_dim = out_dim
        self.convs = ModuleList()

        for i in range(num_layers):
            in_dim = node_feature_dim if i == 0 else hidden_dim
            mlp_node = MLP([in_dim, hidden_dim, hidden_dim], act="ReLU", batch_norm=True)
            
            conv = GINEConv_concat(nn_node=mlp_node, 
                            edge_dim=edge_dim,
                            hidden_dim=hidden_dim,
                            eps=0.0, 
                            train_eps=False)
                            
            self.convs.append(conv)

        self.final_mlp = MLP([hidden_dim, hidden_dim, out_dim], act="ReLU", batch_norm=True)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.f_d, data.edge_index, data.edge_f_o, data.batch
        edge_index, _ = remove_self_loops(edge_index, None)

        # x = self.atom_encoder(x)
        # edge_attr = self.edge_encoder(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.final_mlp(x)

        return x


