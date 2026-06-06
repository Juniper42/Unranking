import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .BaseModel import BaseModel


class GAT(BaseModel):
    """
    Graph Attention Network (GAT) model for recommendation.

    Args:
        num_users: Number of users.
        num_items: Number of items.
        emb_dim: Dimension of the embedding vectors.
        num_layers: Number of GAT layers.
        dropout_rate: Dropout rate.
    """

    def __init__(
        self, num_users, num_items, emb_dim=64, num_layers=2, dropout_rate=0.0
    ):
        super().__init__(num_users, num_items, emb_dim)
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # User/item embedding layers
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        # GAT convolutional layers
        self.convs = nn.ModuleList(
            [
                GATConv(emb_dim, emb_dim, heads=1, concat=False, dropout=dropout_rate)
                for _ in range(num_layers)
            ]
        )

        self._init_weights()

    def _init_weights(self):
        """Initializes model weights."""
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, user_indices=None, item_indices=None, edge_index=None):
        """
        Forward pass of the model.

        Supports two modes:
        1. When `edge_index` is provided, it performs graph convolutions and returns all user and item embeddings.
        2. When `user_indices` and `item_indices` are provided, it computes prediction scores for the pairs.

        Args:
            user_indices: Tensor of user indices.
            item_indices: Tensor of item indices.
            edge_index: The edge index of the graph.

        Returns:
            - A tuple of (user_embeddings, item_embeddings) if `edge_index` is provided.
            - A tensor of prediction scores if `user_indices` and `item_indices` are provided.
        """
        if edge_index is not None:
            return self._forward_gnn(edge_index)

        if user_indices is not None and item_indices is not None:
            user_emb = self.user_emb(user_indices)
            item_emb = self.item_emb(item_indices)
            return torch.sum(user_emb * item_emb, dim=1)

        raise ValueError(
            "Either edge_index or both user_indices and item_indices must be provided."
        )

    def _forward_gnn(self, edge_index):
        """
        Performs GAT graph convolutions.

        Args:
            edge_index: The edge index of the graph.

        Returns:
            A tuple of (user_embeddings, item_embeddings).
        """
        # Initial embeddings
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_embs = [x]

        # Multi-layer graph convolutions
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            all_embs.append(x)

        # Aggregate embeddings from all layers
        final_emb = torch.stack(all_embs, dim=0).mean(dim=0)

        # Split into user and item embeddings
        user_embeddings, item_embeddings = torch.split(
            final_emb, [self.num_users, self.num_items]
        )
        return user_embeddings, item_embeddings
