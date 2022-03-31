from torch._C import device
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import torch


class GraphCls(torch.nn.Module):
    def __init__(self,hidden_channels=256): 
        super(GraphCls,self).__init__()
        torch.manual_seed(1022)
        self.conv1 = gnn.GCNConv(512,hidden_channels)
        self.conv2 = gnn.GCNConv(hidden_channels,hidden_channels)
        self.conv3 = gnn.GCNConv(hidden_channels,hidden_channels)
        #self.pool1 = gnn.SAGPooling(hidden_channels*3,min_score=None)
        self.pool1 = gnn.TopKPooling(hidden_channels*3,ratio=0.8,nonlinearity=torch.sigmoid)
        self.line1 = nn.Linear(hidden_channels*3*2,hidden_channels)
        self.line2 = nn.Linear(hidden_channels,hidden_channels//2)
        self.line3 = nn.Linear(hidden_channels//2,3)
    def forward(self,x,edge_index,edge_attr,batch,get_score=False):
        x1 = F.relu(self.conv1(x,edge_index,edge_attr))
        x2 = F.relu(self.conv2(x1,edge_index,edge_attr))
        x3 = F.relu(self.conv2(x2,edge_index,edge_attr))
        cat_x = torch.cat((x1,x2,x3),dim=1)
        x, edge_index, edge_attr, batch, _, _ = self.pool1(cat_x,edge_index,edge_attr,batch)
        readout = torch.cat((gap(x, batch),gmp(x, batch)), dim=1)
        output = F.relu(self.line1(readout))
        output = F.relu(self.line2(output))
        output = F.dropout(output,p=0,training=self.training)
        output = self.line3(output)
        return output
