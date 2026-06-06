import torch
import torch.nn as nn
import torch.nn.functional as F


class MIAttackModel(nn.Module):
    """
    A simple MLP-based model for Membership Inference Attacks.
    This model tries to predict whether a given user-item interaction was part of the
    training set of the target recommendation model.

    Args:
        input_dim: The dimension of the input features.
        dropout: The dropout rate to prevent overfitting.
    """

    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        hidden1 = max(128, input_dim // 2)
        hidden2 = 64

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using Kaiming Normal (He) initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the attack model.

        Args:
            x: The input feature tensor.

        Returns:
            The output logits from the attack model.
        """
        return self.model(x)
