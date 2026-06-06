import torch
import torch.nn as nn
import torch.nn.functional as F

from .BaseModel import BaseModel


class WMF(BaseModel):
    """
    WMF (Weighted Matrix Factorization) model.
    This implementation is based on the logic from a corresponding TensorFlow version,
    designed for implicit feedback datasets.

    Args:
        num_users: Total number of users.
        num_items: Total number of items.
        emb_dim: Dimensionality of user and item embeddings.
        weight1: Corresponds to 'negative_weight' in the original TF code, used in the custom loss.
        lambda_user: L2 regularization strength for user embeddings.
        lambda_item: L2 regularization strength for item embeddings.
        dropout_rate: Dropout rate for user embeddings.
    """

    def __init__(
        self,
        num_users,
        num_items,
        emb_dim,
        weight1=0.5,
        lambda_user=0.01,
        lambda_item=0.01,
        dropout_rate=0.5,
    ):
        super().__init__(num_users, num_items, emb_dim)
        self.weight1 = weight1
        self.lambda_user = lambda_user
        self.lambda_item = lambda_item

        self.user_emb = nn.Embedding(self.num_users, self.emb_dim)
        # Add a padding embedding for items, which is common in WMF implementations.
        self.item_emb = nn.Embedding(self.num_items + 1, self.emb_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        """Initializes model weights."""
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)
        # Zero out the padding embedding
        with torch.no_grad():
            self.item_emb.weight[self.num_items].zero_()

    def forward(self, user_indices=None, item_indices=None, edge_index=None):
        """
        Computes interaction scores for given user-item pairs.

        Args:
            user_indices: Tensor of user indices.
            item_indices: Tensor of item indices.
            edge_index: Ignored for this non-GNN model.

        Returns:
            A tensor of prediction scores.
        """
        if user_indices is None or item_indices is None:
            raise ValueError("user_indices and item_indices must be provided for WMF.")

        user_emb = self.user_emb(user_indices)
        item_emb = self.item_emb(item_indices)
        user_emb_dropout = self.dropout(user_emb)
        return torch.sum(user_emb_dropout * item_emb, dim=1)

    def get_embeddings(self):
        """
        Returns all user and item embeddings (excluding the padding item embedding).
        """
        return self.user_emb.weight, self.item_emb.weight[: self.num_items]

    def compute_loss(
        self,
        user_batch_embeds_dropout,
        pos_item_batch_embeds_padded,
        pos_item_mask,
        all_user_embeds,
        all_item_embeds_no_pad,
    ):
        """
        Computes the WMF loss, adapted from a reference TensorFlow implementation.

        Args:
            user_batch_embeds_dropout: Embeddings of users in the current batch after dropout.
            pos_item_batch_embeds_padded: Padded embeddings of positive items for users in the batch.
            pos_item_mask: A mask to identify valid (non-padded) positive items.
            all_user_embeds: All user embeddings.
            all_item_embeds_no_pad: All item embeddings, excluding the padding one.

        Returns:
            The total computed loss as a scalar tensor.
        """
        # Loss from positive interactions
        pos_r = torch.einsum(
            "bd,bnd->bn", user_batch_embeds_dropout, pos_item_batch_embeds_padded
        )
        loss_pos_interaction = torch.sum(
            ((1.0 - self.weight1) * torch.square(pos_r) - 2.0 * pos_r) * pos_item_mask
        )

        # Regularization term from the original TF code's loss formulation
        term_V = torch.matmul(all_item_embeds_no_pad.t(), all_item_embeds_no_pad)
        term_U = torch.matmul(all_user_embeds.t(), all_user_embeds)
        loss_reg_term1 = self.weight1 * torch.sum(term_U * term_V)

        # Standard L2 regularization on all embeddings
        l2_loss_user = torch.norm(all_user_embeds, 2).pow(2) / 2.0
        l2_loss_item = torch.norm(all_item_embeds_no_pad, 2).pow(2) / 2.0

        total_loss = (
            loss_pos_interaction
            + loss_reg_term1
            + self.lambda_user * l2_loss_user
            + self.lambda_item * l2_loss_item
        )
        return total_loss
