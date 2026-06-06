import torch

from models import GAT, GMF, MLP, WMF, GraphSage, LightGCN, NeuMF


def get_model(args, num_users, num_items):
    """
    Factory function to create and return a recommendation model instance based on the provided arguments.

    Args:
        args (dict): A dictionary of arguments containing model specifications.
        num_users (int): The total number of users.
        num_items (int): The total number of items.

    Returns:
        An instance of the specified recommendation model.

    Raises:
        ValueError: If the specified backbone is not supported.
    """
    backbone = args.get("backbone", "lightgcn")
    emb_dim = args.get("emb_dim", 64)
    num_layers = args.get("num_layers", 2)
    dropout_rate = args.get("dropout_rate", 0.0)

    if backbone == "lightgcn":
        return LightGCN(num_users, num_items, emb_dim=emb_dim, num_layers=num_layers)
    elif backbone == "gat":
        return GAT(
            num_users,
            num_items,
            emb_dim=emb_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )
    elif backbone == "graphsage":
        return GraphSage(
            num_users,
            num_items,
            emb_dim=emb_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )
    elif backbone == "mlp":
        return MLP(
            num_users,
            num_items,
            emb_dim=emb_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )
    elif backbone == "gmf":
        return GMF(num_users, num_items, emb_dim=emb_dim, dropout_rate=dropout_rate)
    elif backbone == "neumf":
        return NeuMF(
            num_users,
            num_items,
            emb_dim=emb_dim,
            layers=args.get("layers", [128, 64, 32, 16]),
            dropout_rate=dropout_rate,
        )
    elif backbone == "wmf":
        return WMF(num_users, num_items, emb_dim=emb_dim, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")


def determine_is_gnn(backbone_name):
    """
    Determines if a model is a Graph Neural Network (GNN) based on its name.

    Args:
        backbone_name (str): The name of the model backbone.

    Returns:
        bool: True if the model is identified as a GNN, False otherwise.
    """
    if not backbone_name:
        return False

    known_gnn_keywords = [
        "gcn",
        "gat",
        "gin",
        "graphsage",
        "lightgcn",
        "sgc",
        "appnp",
        "tagcn",
        "chebnet",
        "sage",
        "rgcn",
    ]
    model_name_lower = backbone_name.lower()
    return any(keyword in model_name_lower for keyword in known_gnn_keywords)
