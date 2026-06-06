import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from .BaseModel import BaseModel


class GraphSage(BaseModel):
    """
    GraphSAGE model for recommendation.

    Args:
        num_users: Number of users.
        num_items: Number of items.
        emb_dim: Dimension of the embedding vectors.
        num_layers: Number of GraphSAGE layers.
        dropout_rate: Dropout rate.
    """

    def __init__(
        self, num_users, num_items, emb_dim=64, num_layers=2, dropout_rate=0.0
    ):
        super().__init__(num_users, num_items, emb_dim)
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        self.convs = nn.ModuleList(
            [SAGEConv(emb_dim, emb_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout_rate)

        self._init_weights()

        self._cached_embeddings = None
        self._cached_edge_index = None

    def clear_cache(self):
        """Clears the cached embeddings."""
        self._cached_embeddings = None
        self._cached_edge_index = None
        super().clear_cache()

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
            if (
                self._cached_embeddings is not None
                and not self.training
                and self._cached_edge_index is not None
                and torch.equal(edge_index, self._cached_edge_index)
            ):
                return self._cached_embeddings

            user_embs, item_embs = self._forward_gnn(edge_index)

            if not self.training:
                self._cached_embeddings = (user_embs, item_embs)
                self._cached_edge_index = edge_index
            return user_embs, item_embs

        if user_indices is not None and item_indices is not None:
            user_emb = self.user_emb(user_indices)
            item_emb = self.item_emb(item_indices)
            return torch.sum(user_emb * item_emb, dim=1)

        raise ValueError(
            "Either edge_index or both user_indices and item_indices must be provided."
        )

    def _forward_gnn(self, edge_index):
        """
        Performs GraphSAGE graph convolutions.

        Args:
            edge_index: The edge index of the graph.

        Returns:
            A tuple of (user_embeddings, item_embeddings).
        """
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_embs = [x]

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            all_embs.append(x)

        final_emb = torch.stack(all_embs, dim=0).mean(dim=0)

        user_embeddings, item_embeddings = torch.split(
            final_emb, [self.num_users, self.num_items]
        )
        return user_embeddings, item_embeddings
