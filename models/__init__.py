# models/__init__.py

"""
This file makes the models available as a package.
It defines what is imported when 'from models import *' is used.
"""

# Base model
from .BaseModel import BaseModel

# GNN-based models
from .GAT import GAT

# Non-GNN models
from .GMF import GMF
from .GraphSage import GraphSage
from .LightGCN import LightGCN, LightGCNConv

# Attack model
from .MIAttackModel import MIAttackModel
from .MLP import MLP
from .NeuMF import NeuMF
from .WMF import WMF

# The __all__ variable defines the public API of the models package.
# These are the names that will be imported when a client uses 'from models import *'.
__all__ = [
    # Base Model
    "BaseModel",
    # GNN Models
    "GAT",
    "GraphSage",
    "LightGCN",
    "LightGCNConv",
    # Non-GNN Models
    "GMF",
    "MLP",
    "NeuMF",
    "WMF",
    # Attack Model
    "MIAttackModel",
]
