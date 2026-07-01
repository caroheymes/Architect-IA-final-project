import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class SpatioTemporalGCN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network (ST-GRU-GNN).

    Note: While named 'SpatioTemporalGCN' (STGCN) for simplicity in the codebase,
    this architecture differs from the original STGCN by Yu et al. (2018) which uses 1D causal
    temporal convolutions with GLU gates.
    Instead, this is a hybrid recurrent-spatial model ('GRU + GCN with skip connections')
    which combines a temporal recurrent layer (nn.GRU) with spatial graph convolutions (GCNConv)
    and residual skip connections. This design is highly robust to noise and irregular sampling
    inherent in Lyon's real-world traffic data.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.0):
        """Initialise les sous-modules du modèle ST-GRU-GNN.

        Args:
            in_channels (int): Nb de features d'entrée par pas de temps
                (= 5 dans `build_sliding_dataset` : speed + 4 cycliques).
            hidden_channels (int): Taille de l'état caché partagé entre GRU et GCN.
            out_channels (int): Nb d'horizons prédits (= `len(horizons)`).
            dropout (float): Taux de dropout pour régulariser les couches spatiales.
        """
        super().__init__()
        # Encodeur temporel : GRU traitant la séquence par nœud.
        # Le batching PyG aplatit le batch en `[B*N, SEQ_LEN, in_channels]`,
        # donc une seule GRU "multi-nœuds" traite tout le batch.
        self.temporal_gru = nn.GRU(input_size=in_channels, hidden_size=hidden_channels, num_layers=1, batch_first=True)
        # Encodeur spatial : deux GCN avec self-loops (ajoutés dans le dataset).
        self.spatial_gcn1 = GCNConv(hidden_channels, hidden_channels)
        self.spatial_gcn2 = GCNConv(hidden_channels, hidden_channels)

        # Couche fully-connected de régression multi-horizon.
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        """Passe forward du modèle.

        Étapes :
          1. **Encodage temporel** : la GRU consomme la fenêtre
             `[B*N, SEQ_LEN, in_channels]` et on ne garde que le dernier
             état caché `h_temp` (résumé de la dynamique temporelle).
          2. **Convolution spatiale 1** : GCN propage l'info sur le graphe
             routier, suivi d'une LeakyReLU et d'une **skip connection**
             (`+ h_temp`) pour stabiliser la backprop.
          3. **Convolution spatiale 2** : idem avec un skip depuis
             `h_space1`.
          4. **Tête de régression** : `Linear(hidden, out_channels)`
             produit la prédiction `[B*N, out_channels]`.

        Args:
            x (torch.Tensor): Entrée `[B*N, SEQ_LEN, in_channels]`.
            edge_index (torch.Tensor): `edge_index` PyG de shape `[2, 2*E + N]`.

        Returns:
            torch.Tensor: Prédictions de shape `[B*N, out_channels]`.
        """
        # x shape: [B * N, SEQ_LEN, in_channels]
        gru_out, _ = self.temporal_gru(x)
        h_temp = gru_out[:, -1, :]  # Keep last step hidden state: [B * N, hidden_channels]

        # Spatial convolution 1 with Skip Connection
        h_space1 = self.spatial_gcn1(h_temp, edge_index)
        h_space1 = self.relu(h_space1)
        h_space1 = self.dropout(h_space1) + h_temp

        # Spatial convolution 2 with Skip Connection
        h_space2 = self.spatial_gcn2(h_space1, edge_index)
        h_space2 = self.relu(h_space2)
        h_space2 = self.dropout(h_space2) + h_space1

        return self.fc(h_space2)
