import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base class for recommendation system models.
    Defines a common interface for all recommendation models to ensure API consistency.
    """

    def __init__(self, num_users, num_items, emb_dim=64):
        """
        Initializes the base recommendation model.

        Args:
            num_users: The number of users.
            num_items: The number of items.
            emb_dim: The dimension of the embedding vectors.
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim

        # User and item embedding layers - subclasses may redefine these.
        self.user_emb = None
        self.item_emb = None

    def forward(self, user_indices=None, item_indices=None, edge_index=None):
        """
        Forward pass of the model.

        This method should be implemented by all subclasses. It supports two modes:
        1. When `edge_index` is provided (for GNNs), it should return the embeddings for all users and items.
        2. When `user_indices` and `item_indices` are provided, it should return the prediction scores
           for the given user-item pairs.

        Args:
            user_indices: Tensor of user indices.
            item_indices: Tensor of item indices.
            edge_index: The edge index of the graph for GNN-based models.

        Returns:
            Either a tuple of (user_embeddings, item_embeddings) or a tensor of prediction scores.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")

    def get_embeddings(self):
        """
        Retrieves the user and item embeddings.

        Returns:
            A tuple containing (user_embeddings, item_embeddings).
        """
        if self.user_emb is not None and self.item_emb is not None:
            return self.user_emb.weight, self.item_emb.weight
        else:
            raise NotImplementedError(
                "This model does not have standard user/item embedding layers. "
                "Override the get_embeddings method."
            )

    def clear_cache(self):
        """Clears any cached data, such as embeddings in GNN models."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _init_weights(self):
        """Initializes the model's weights."""
        if hasattr(self, "user_emb") and self.user_emb is not None:
            nn.init.xavier_uniform_(self.user_emb.weight)
        if hasattr(self, "item_emb") and self.item_emb is not None:
            nn.init.xavier_uniform_(self.item_emb.weight)
