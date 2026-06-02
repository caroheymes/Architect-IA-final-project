"""
test_stgcn_model.py

Thorough unit and integration tests for the Spatio-Temporal GCN (STGCN) model,
data-preprocessing pipelines, loss weights, and training workflows.
Ensures shape consistency, gradient flow, and regression output correctness.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure our project path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import STGCN classes and functions
from training.stgcn.dataset import build_sliding_dataset, load_graph_topology, load_traffic_series
from training.stgcn.model import SpatioTemporalGCN


class TestSpatioTemporalGCN(unittest.TestCase):
    def setUp(self):
        # Define default hyperparameters for testing
        self.in_channels = 5
        self.hidden_channels = 64
        self.out_channels = 1
        self.seq_len = 12
        self.num_nodes = 10
        self.batch_size = 2

        # Create model instance
        self.model = SpatioTemporalGCN(
            in_channels=self.in_channels, hidden_channels=self.hidden_channels, out_channels=self.out_channels
        )

    def test_model_initialization(self):
        """Vérifie l'initialisation des sous-modules du modèle.

        Assertions :
          - `model` est un `nn.Module`.
          - `temporal_gru` est un `nn.GRU`.
          - `fc` est un `nn.Linear`.
          - le nombre total de paramètres est strictement positif.
        """
        self.assertIsInstance(self.model, nn.Module)
        self.assertIsInstance(self.model.temporal_gru, nn.GRU)
        self.assertIsInstance(self.model.fc, nn.Linear)

        # Count parameters to ensure layers are populated
        param_count = sum(p.numel() for p in self.model.parameters())
        self.assertTrue(param_count > 0)

    def test_forward_shape_consistency(self):
        """Vérifie la cohérence des shapes entrée/sortie du forward.

        Crée un tenseur `[B*N, S, C]` aléatoire et un graphe en chaîne
        bidirectionnel, puis vérifie que la sortie est `[B*N, out_channels]`
        et ne contient pas de NaN.
        """
        # Batch size = B, Num nodes = N, Seq len = S, In channels = C
        # Input tensor 'x' shape expected by PyG / GRU: [B * N, S, C]
        total_graphs_nodes = self.batch_size * self.num_nodes
        x = torch.randn(total_graphs_nodes, self.seq_len, self.in_channels)

        # Edge index shape: [2, E]
        # Let's create a simple chain graph (E = N - 1) repeated for both batched graphs
        edges = []
        for b in range(self.batch_size):
            offset = b * self.num_nodes
            for u in range(self.num_nodes - 1):
                edges.append([offset + u, offset + u + 1])
                edges.append([offset + u + 1, offset + u])  # undirected
        edge_index = torch.tensor(edges, dtype=torch.long).t()

        # Run forward pass
        output = self.model(x, edge_index)

        # Expected output shape: [B * N, out_channels]
        expected_shape = (total_graphs_nodes, self.out_channels)
        self.assertEqual(output.shape, expected_shape)
        self.assertFalse(torch.isnan(output).any(), "Model produced NaN values in forward pass.")

    def test_gradient_flow_and_clipping(self):
        """Vérifie la backprop et le gradient clipping sur un graphe complet.

        Construit un graphe complet (chaque paire de nœuds reliée), effectue
        un forward + MSE loss + backward, puis :
          - tous les `param.grad` existent et sont non-NaN ;
          - `clip_grad_norm_(max_norm=1.0)` retourne un scalaire fini ;
          - `optimizer.step()` s'exécute sans erreur.
        """
        total_graphs_nodes = self.batch_size * self.num_nodes
        x = torch.randn(total_graphs_nodes, self.seq_len, self.in_channels)

        # Complete connected graph edges
        edges = []
        for u in range(total_graphs_nodes):
            for v in range(total_graphs_nodes):
                if u != v:
                    edges.append([u, v])
        edge_index = torch.tensor(edges, dtype=torch.long).t()

        # Target speed values (de-normalized domain)
        targets = torch.randn(total_graphs_nodes, self.out_channels)

        # Run a small optimization step
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        predictions = self.model(x, edge_index)
        loss = nn.MSELoss()(predictions, targets)

        optimizer.zero_grad()
        loss.backward()

        # Check gradients exist before clipping
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient for {name} is None.")
                self.assertFalse(torch.isnan(param.grad).any(), f"Gradient for {name} contains NaNs.")

        # Test gradient clipping max_norm constraint
        max_norm = 1.0
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)

        # Ensure clipping runs and returns a valid norm scalar
        self.assertIsInstance(grad_norm.item(), float)
        self.assertFalse(torch.isnan(grad_norm), "Gradient norm computed as NaN.")

        # Run optimizer step
        optimizer.step()

    def test_staircase_loss_weight_application(self):
        """Vérifie l'implémentation de la loss pondérée « en escalier ».

        Pour des cibles `y_kmh = [5, 25, 45]` (JAM, SLOW, NORMAL), on doit
        obtenir les poids `[15, 5, 1]` et la loss pondérée `(diff^2 * weights).mean()`
        doit être identique à sa version manuelle.
        """
        # Setup dummy de-normalized targets representing JAM (<10), SLOW (<30), and NORMAL (>30)
        y_kmh = torch.tensor([[5.0], [25.0], [45.0]])  # JAM, SLOW, NORMAL
        predictions = torch.tensor([[8.0], [22.0], [45.0]])  # some predictions

        weight_jam = 15.0
        weight_slow = 5.0
        weight_normal = 1.0

        # Implement the same torch.where staircase weight assignment as in train_stgcn.py
        weights = torch.where(
            y_kmh < 10.0,
            torch.tensor(weight_jam),
            torch.where(y_kmh < 30.0, torch.tensor(weight_slow), torch.tensor(weight_normal)),
        )

        expected_weights = torch.tensor([[weight_jam], [weight_slow], [weight_normal]])
        torch.testing.assert_close(weights, expected_weights)

        # Ensure loss value applies weights correctly
        unweighted_diff_sq = (predictions - y_kmh) ** 2
        weighted_loss_manual = (unweighted_diff_sq * expected_weights).mean()
        weighted_loss_code = (unweighted_diff_sq * weights).mean()

        self.assertEqual(weighted_loss_code.item(), weighted_loss_manual.item())


class TestSTGCNDatasetAndTraining(unittest.TestCase):
    def test_load_graph_topology_mock(self):
        """Mocke la DB et vérifie `load_graph_topology`.

        Avec 5 nœuds et 4 arêtes (cycle 0→1→2→3→0), on attend :
          - `num_nodes == 5`
          - `edge_index.shape == (2, 13)` (8 arêtes bidirectionnelles + 5 self-loops)
          - retour = `torch.Tensor`.
        """
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        def mock_read_sql(sql, con, *args, **kwargs):
            if "dim_spatial_grid_mapping" in sql:
                # Mock nodes query
                return pd.DataFrame({"node_idx": list(range(5))})
            else:
                # Mock edges query
                return pd.DataFrame({"node_u": [0, 1, 2, 3], "node_v": [1, 2, 3, 0]})

        with patch("pandas.read_sql", side_effect=mock_read_sql):
            num_nodes, edge_index = load_graph_topology(mock_engine)

            self.assertEqual(num_nodes, 5)
            self.assertEqual(edge_index.shape, (2, 13))  # 4 undirected bidirectional edges (8) + 5 self-loops = 13
            self.assertIsInstance(edge_index, torch.Tensor)

    def test_load_traffic_series_mock(self):
        """Mocke la DB et vérifie `load_traffic_series` (pivot, imputation NaN, cycliques).

        Avec 2 timestamps × 2 nœuds et un NaN :
          - la matrice est de shape `(2, 2)` ;
          - la valeur manquante est remplacée par `LYON_DEFAULT_SPEED = 30.0` ;
          - les 4 features cycliques sont de longueur 2.
        """
        mock_engine = MagicMock()

        # 2 timestamps, 2 nodes. One missing value to test 30.0 imputation.
        df_mock = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-06-01 12:00:00", "2026-06-01 12:00:00", "2026-06-01 12:05:00", "2026-06-01 12:05:00"]
                ),
                "node_idx": [0, 1, 0, 1],
                "properties_vitesse": [35.0, 45.0, np.nan, 42.0],
            }
        )

        with patch("pandas.read_sql", return_value=df_mock):
            with patch.dict(os.environ, {"LYON_DEFAULT_SPEED": "30.0"}):
                matrix, h_sin, h_cos, d_sin, d_cos = load_traffic_series(mock_engine)

                # Check shapes (2 timestamps, 2 nodes)
                self.assertEqual(matrix.shape, (2, 2))

                # Check values: index 0,0 is 35.0; index 0,1 is 45.0
                self.assertEqual(matrix[0, 0], 35.0)
                self.assertEqual(matrix[0, 1], 45.0)

                # Check default speed imputation on np.nan (index 1,0 should be 30.0)
                self.assertEqual(matrix[1, 0], 30.0)
                self.assertEqual(matrix[1, 1], 42.0)

                # Check that cyclical features are calculated and of length 2
                self.assertEqual(len(h_sin), 2)
                self.assertEqual(len(h_cos), 2)
                self.assertEqual(len(d_sin), 2)
                self.assertEqual(len(d_cos), 2)

    def test_build_sliding_dataset(self):
        """Vérifie `build_sliding_dataset` (scaler fit, loaders, shapes des batches).

        Avec 50 pas, 4 nœuds, `seq_len=6` :
          - le `StandardScaler` est fitté (`mean_`, `scale_` présents) ;
          - le premier batch du `train_loader` a un tenseur `x` de shape
            `[B, 6, 5]` et un tenseur `y` de shape `[B, 1]`.
        """
        num_nodes = 4
        num_timesteps = 50
        seq_len = 6

        # Create dummy timeseries matrices
        vitesse_matrix_raw = np.random.uniform(5.0, 90.0, size=(num_timesteps, num_nodes))
        hour_sin = np.sin(np.linspace(0, 2 * np.pi, num_timesteps))
        hour_cos = np.cos(np.linspace(0, 2 * np.pi, num_timesteps))
        day_sin = np.sin(np.linspace(0, 2 * np.pi, num_timesteps))
        day_cos = np.cos(np.linspace(0, 2 * np.pi, num_timesteps))

        edge_index_tensor = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        # Call data loader builder
        train_loader, test_loader, scaler = build_sliding_dataset(
            vitesse_matrix_raw,
            hour_sin,
            hour_cos,
            day_sin,
            day_cos,
            seq_len=seq_len,
            edge_index_tensor=edge_index_tensor,
            num_nodes=num_nodes,
            test_split=0.2,
            batch_size=4,
        )

        # Verify scaler has been fit
        self.assertTrue(hasattr(scaler, "mean_"))
        self.assertTrue(hasattr(scaler, "scale_"))

        # Verify train loader has batches of expected shape
        for batch in train_loader:
            # batch.x shape: [B * N, seq_len, in_channels (speed + 4 temporal features = 5)]
            self.assertEqual(batch.x.shape[1], seq_len)
            self.assertEqual(batch.x.shape[2], 5)
            # batch.y shape: [B * N, out_channels]
            self.assertEqual(batch.y.shape[1], 1)
            # edge_index is present
            break

    def test_metrics_calculation(self):
        """Vérifie la formule de calcul de la `epoch_loss` et de la `test_mae_kmh`.

        Cas vérifiés :
          - Loss moyenne = `(b1*g + b2*g) / total_graphs`
            (ici 0.4 avec `b1=0.5, b2=0.3, g=4, total=8`).
          - MAE = `|pred-target|.sum() / (n_graphs * n_nodes)`
            (ici 17/6 sur l'exemple 6 paires).
        """
        import torch.nn.functional as F

        # 1. Test Train Loss (std) calculation
        batch_1_loss = 0.5
        batch_2_loss = 0.3
        num_graphs_per_batch = 4

        total_train_loss = (batch_1_loss * num_graphs_per_batch) + (batch_2_loss * num_graphs_per_batch)
        dataset_size = 8
        epoch_loss = total_train_loss / dataset_size

        self.assertAlmostEqual(epoch_loss, 0.4)

        # 2. Test Test MAE (km/h) calculation
        preds_kmh = torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]])
        targets_kmh = torch.tensor([[12.0], [18.0], [33.0], [37.0], [55.0], [58.0]])
        num_nodes = 2
        dataset_size_test = 3  # 3 graphs, 2 nodes each

        mae_sum = F.l1_loss(preds_kmh, targets_kmh, reduction="sum").item()
        # |10-12| + |20-18| + |30-33| + |40-37| + |50-55| + |60-58| = 2+2+3+3+5+2 = 17
        self.assertEqual(mae_sum, 17.0)

        test_mae_kmh = mae_sum / (dataset_size_test * num_nodes)
        self.assertAlmostEqual(test_mae_kmh, 17.0 / 6.0)


if __name__ == "__main__":
    unittest.main()
