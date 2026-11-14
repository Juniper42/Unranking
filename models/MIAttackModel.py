import torch.nn as nn


class MIAttackModel(nn.Module):
    """Three-layer MLP used by the membership-inference probe.

    The paper (Section 5.1.6) specifies a three-layer MLP with hidden
    dimensions 64 and 32 and ReLU activations.  The probe is trained on the
    pre-revision model outputs and then frozen to evaluate every revised
    model, so the architecture is intentionally small.

    Args:
        input_dim: Dimension of the input feature vector (eight in the paper).
        dropout: Optional dropout on the hidden layers; the paper does not use
            dropout, so the default is 0.0.
    """

    def __init__(self, input_dim, dropout=0.0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)
