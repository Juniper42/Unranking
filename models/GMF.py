import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseModel import BaseModel


class GMF(BaseModel):
    """
    Generalized Matrix Factorization (GMF) model for recommendation.
    This implementation enhances the basic GMF by adding optional non-linear layers
    to improve its expressive power.

    Args:
        num_users: Number of users.
        num_items: Number of items.
        emb_dim: Dimension of the embedding vectors.
        dropout_rate: Dropout rate.
        hidden_layers: A list of integers specifying the size of the hidden layers. Defaults to [64].
    """

    def __init__(
        self, num_users, num_items, emb_dim=64, dropout_rate=0.1, hidden_layers=[64]
    ):
        super().__init__(num_users, num_items, emb_dim)
        self.dropout_rate = dropout_rate

        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        layers = []
        input_dim = emb_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.prediction_layers = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        """Initializes model weights using Xavier uniform distribution."""
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        for m in self.prediction_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_indices=None, item_indices=None, edge_index=None):
        """
        Computes the prediction scores for given user-item pairs.

        Args:
            user_indices: Tensor of user indices.
            item_indices: Tensor of item indices.
            edge_index: Ignored for this non-GNN model.

        Returns:
            A tensor of prediction scores.
        """
        if user_indices is None or item_indices is None:
            raise ValueError("user_indices and item_indices must be provided for GMF.")

        user_embedding = self.user_emb(user_indices)
        item_embedding = self.item_emb(item_indices)

        # Element-wise product of user and item embeddings
        vector = user_embedding * item_embedding

        prediction = self.prediction_layers(vector)
        return prediction.squeeze(-1)
