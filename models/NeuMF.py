import torch
import torch.nn as nn

from .BaseModel import BaseModel


class NeuMF(BaseModel):
    """
    Neural Matrix Factorization (NeuMF) model for recommendation.
    This model combines Generalized Matrix Factorization (GMF) and a Multi-Layer Perceptron (MLP)
    to capture both linear and non-linear user-item interactions.

    Args:
        num_users: Number of users.
        num_items: Number of items.
        emb_dim: Dimension of the GMF embedding vectors.
        layers: A list defining the MLP layer structure, e.g., [128, 64, 32].
        dropout_rate: Dropout rate for the MLP layers.
    """

    def __init__(
        self,
        num_users,
        num_items,
        emb_dim=64,
        layers=[128, 64, 32, 16],
        dropout_rate=0.2,
    ):
        super().__init__(num_users, num_items, emb_dim)

        # GMF specific embeddings
        self.user_gmf_emb = nn.Embedding(num_users, emb_dim)
        self.item_gmf_emb = nn.Embedding(num_items, emb_dim)

        # MLP specific embeddings
        mlp_emb_dim = layers[0] // 2
        self.user_mlp_emb = nn.Embedding(num_users, mlp_emb_dim)
        self.item_mlp_emb = nn.Embedding(num_items, mlp_emb_dim)

        # MLP layers
        self.mlp_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.mlp_layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout_rate))

        # Prediction layer: concatenates the outputs of GMF and MLP
        self.predict_layer = nn.Linear(emb_dim + layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        """Initializes model weights."""
        nn.init.normal_(self.user_gmf_emb.weight, std=0.01)
        nn.init.normal_(self.item_gmf_emb.weight, std=0.01)
        nn.init.normal_(self.user_mlp_emb.weight, std=0.01)
        nn.init.normal_(self.item_mlp_emb.weight, std=0.01)

        for m in self.mlp_layers:
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
            raise ValueError(
                "user_indices and item_indices must be provided for NeuMF."
            )

        # GMF part
        user_gmf = self.user_gmf_emb(user_indices)
        item_gmf = self.item_gmf_emb(item_indices)
        gmf_vector = user_gmf * item_gmf

        # MLP part
        user_mlp = self.user_mlp_emb(user_indices)
        item_mlp = self.item_mlp_emb(item_indices)
        mlp_vector = torch.cat([user_mlp, item_mlp], dim=-1)
        for layer in self.mlp_layers:
            mlp_vector = layer(mlp_vector)

        # Concatenate GMF and MLP vectors
        vector = torch.cat([gmf_vector, mlp_vector], dim=-1)

        # Final prediction
        prediction = self.predict_layer(vector)
        return prediction.squeeze(-1)

    def get_embeddings(self):
        """
        Returns the GMF embeddings as the primary embeddings for this model.
        Note: NeuMF has separate embeddings for GMF and MLP parts. This method returns
        the GMF embeddings for general-purpose use (e.g., in evaluation).
        """
        return self.user_gmf_emb.weight, self.item_gmf_emb.weight
