import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class SpatioTemporalGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network (STGCN).
    Combines a temporal recurrence layer (GRU) with spatial graph convolutions (GCNConv)
    and residual skip connections to model spatial-temporal dynamics of traffic speeds.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SpatioTemporalGCN, self).__init__()
        # Temporal encoder: GRU process per-node sequence
        self.temporal_gru = nn.GRU(
            input_size=in_channels, 
            hidden_size=hidden_channels, 
            num_layers=1, 
            batch_first=True
        )
        # Spatial encoder: GCN with self-loops
        self.spatial_gcn1 = GCNConv(hidden_channels, hidden_channels)
        self.spatial_gcn2 = GCNConv(hidden_channels, hidden_channels)
        
        # Fully connected regression layer
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        # x shape: [B * N, SEQ_LEN, in_channels]
        gru_out, _ = self.temporal_gru(x)
        h_temp = gru_out[:, -1, :] # Keep last step hidden state: [B * N, hidden_channels]
        
        # Spatial convolution 1 with Skip Connection
        h_space1 = self.spatial_gcn1(h_temp, edge_index)
        h_space1 = self.relu(h_space1) + h_temp
        
        # Spatial convolution 2 with Skip Connection
        h_space2 = self.spatial_gcn2(h_space1, edge_index)
        h_space2 = self.relu(h_space2) + h_space1
        
        return self.fc(h_space2)
