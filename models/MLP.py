import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseModel import BaseModel


class MLP(BaseModel):
    """
    A simple Multi-Layer Perceptron (MLP) model for recommendation tasks.

    Args:
        num_users: Number of users.
        num_items: Number of items.
        emb_dim: Dimension of the embedding vectors.
        num_layers: Number of hidden layers.
        hidden_size: Size of the hidden layers.
        dropout_rate: Dropout rate for hidden layers.
        input_drop: Dropout rate for the input layer.
        residual: Whether to use residual connections.
        activation: The activation function to use.
    """

    def __init__(
        self,
        num_users,
        num_items,
        emb_dim=64,
        num_layers=3,
        hidden_size=64,
        dropout_rate=0.2,
        input_drop=0.1,
        residual=True,
        activation=F.relu,
    ):
        super().__init__(num_users, num_items, emb_dim)
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.input_drop = nn.Dropout(input_drop)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.residual = residual

        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        # Build MLP layers
        in_size = emb_dim * 2
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_hidden = in_size if i == 0 else hidden_size
            self.linears.append(nn.Linear(in_hidden, hidden_size))
            if i < num_layers - 1:
                self.norms.append(nn.BatchNorm1d(hidden_size))

        self.predict_layer = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self):
        """Initializes model weights."""
        super()._init_weights()
        for m in self.linears:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.xavier_uniform_(self.predict_layer.weight)
        nn.init.constant_(self.predict_layer.bias, 0)

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
            raise ValueError("user_indices and item_indices must be provided for MLP.")

        user_embedding = self.user_emb(user_indices)
        item_embedding = self.item_emb(item_indices)

        vector = torch.cat([user_embedding, item_embedding], dim=-1)
        vector = self.input_drop(vector)

        h_last = None
        for i in range(self.num_layers):
            h = self.linears[i](vector)
            if self.residual and i > 0 and h_last is not None:
                h = h + h_last
            h_last = h

            if i < self.num_layers - 1:
                h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)
            vector = h

        prediction = self.predict_layer(vector)
        return prediction.squeeze(-1)
