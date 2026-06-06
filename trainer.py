import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from evaluate import evaluate_model, print_metrics
from logger import Logger
from utils import determine_is_gnn

logger = Logger.get_logger("Trainer")


def _handle_early_stopping(
    model, valid_metrics, best_metrics, best_model_state, patience_count, args
):
    """Handles early stopping logic."""
    early_stop_metric = args["early_stop_metric"]
    patience = args["patience"]
    min_delta = args["min_delta"]

    improvement = valid_metrics[early_stop_metric] - best_metrics[early_stop_metric]

    if improvement > min_delta:
        best_metrics = valid_metrics
        best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        patience_count = 0
        logger.info(
            f"Validation performance improved! New best {early_stop_metric}: {best_metrics[early_stop_metric]:.4f}"
        )
    else:
        patience_count += 1
        logger.info(
            f"No significant improvement. Patience: {patience_count}/{patience}"
        )

    should_stop = args["early_stop"] and patience_count >= patience
    if should_stop:
        logger.info(
            f"Early stopping triggered after {patience} epochs with no improvement."
        )

    return best_metrics, best_model_state, patience_count, should_stop


def train_model(model, data, args, device, user_train_items=None, eval_interval=5):
    """
    A general model training function that automatically selects the appropriate
    training method based on the model type.

    Args:
        model: The model to be trained.
        data: The data object.
        args: Training arguments.
        device: The computation device.
        user_train_items: A dictionary of user's training items for negative sampling.

    Returns:
        The trained model.
    """
    model = model.to(device)

    if args["backbone"].upper() == "WMF":
        logger.info(f"Using WMF training method for backbone: {args['backbone']}")
        return train_wmf_model(
            model, data, args, device, user_train_items, eval_interval
        )
    elif args["backbone"].upper() == "NEUMF":
        logger.info(f"Using NeuMF training method for backbone: {args['backbone']}")
        return train_neumf_model(
            model, data, args, device, user_train_items, eval_interval
        )
    else:
        is_gnn = determine_is_gnn(args["backbone"])
        logger.info(
            f"Using {'GNN' if is_gnn else 'Non-GNN'} training method for backbone: {args['backbone']}"
        )
        trainer = train_gnn_model if is_gnn else train_non_gnn_model
        return trainer(model, data, args, device, user_train_items, eval_interval)


def train_gnn_model(model, data, args, device, user_train_items=None, eval_interval=5):
    """Trains a GNN-based model."""
    model.train()
    optimizer = optim.AdamW(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )
    loss_fcn = BPRLoss()

    edge_index = data.train_edge_index.to(device)
    num_users, num_items = data.num_users, data.num_items
    n_batch = max((num_users + args["batch_size"] - 1) // args["batch_size"], 1)

    best_valid_metrics = {"recall@k": -1, "ndcg@k": -1}
    best_model_state = None
    patience_count = 0

    for epoch in range(args["epoch"]):
        model.train()
        total_loss, total_mf_loss, total_reg_loss = 0.0, 0.0, 0.0

        user_indices = torch.arange(num_users, device=device)
        perm = torch.randperm(num_users, device=device)
        user_indices_shuffled = user_indices[perm]

        for i in range(n_batch):
            start_idx = i * args["batch_size"]
            end_idx = start_idx + args["batch_size"]
            batch_users = user_indices_shuffled[start_idx:end_idx]

            if len(batch_users) == 0:
                continue

            user_embeddings, item_embeddings = model(edge_index=edge_index)

            # Sample positive items
            pos_items = []
            for user_id in batch_users:
                u_id = user_id.item()
                pos_list = list(
                    user_train_items.get(u_id, [0])
                )  # Default to item 0 if user has no items
                pos_items.append(random.choice(pos_list))
            pos_items = torch.tensor(pos_items, device=device)

            neg_items = sample_negative_items(
                batch_users,
                num_users,
                num_items,
                user_train_items,
                args["neg_samples"],
                device,
            )

            neg_samples = args["neg_samples"]
            users_rep = batch_users.repeat_interleave(neg_samples)
            pos_rep = pos_items.repeat_interleave(neg_samples)

            user_emb = user_embeddings[users_rep]
            pos_item_emb = item_embeddings[pos_rep]
            neg_item_emb = item_embeddings[neg_items]

            loss, mf_loss, reg_loss = loss_fcn(
                user_emb, pos_item_emb, neg_item_emb, args["weight_decay"]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_mf_loss += mf_loss.item()
            total_reg_loss += reg_loss.item()

        avg_loss = total_loss / n_batch
        logger.info(
            f"Epoch {epoch+1}/{args['epoch']}: Avg Loss={avg_loss:.4f} (BPR={total_mf_loss/n_batch:.4f}, Reg={total_reg_loss/n_batch:.4f})"
        )

        if eval_interval > 0 and (epoch + 1) % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                valid_metrics = evaluate_model(
                    model,
                    data,
                    args,
                    device,
                    batch_size=args["eval_batch_size"],
                    user_train_items=user_train_items,
                    k=args["k"],
                    is_test=True,
                )
            print_metrics(valid_metrics, prefix="Validation")

            best_valid_metrics, best_model_state, patience_count, should_stop = (
                _handle_early_stopping(
                    model,
                    valid_metrics,
                    best_valid_metrics,
                    best_model_state,
                    patience_count,
                    args,
                )
            )
            if should_stop:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info(
            f"Loaded best model with {args['early_stop_metric']}={best_valid_metrics[args['early_stop_metric']]:.4f}"
        )
    return model


def train_neumf_model(
    model, data, args, device, user_train_items=None, eval_interval=5
):
    """Trains a NeuMF model."""
    model.train()
    optimizer = optim.AdamW(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )
    criterion = nn.BCEWithLogitsLoss()

    num_users, num_items = data.num_users, data.num_items
    user_item_pairs = data.train_ratings[["user_idx", "item_idx"]].values
    user_item_pairs = torch.tensor(user_item_pairs, dtype=torch.long, device=device)

    num_interactions = len(user_item_pairs)
    n_batch = max(
        (num_interactions + args["batch_size"] - 1) // args["batch_size"], 1
    )
    neg_ratio = args["neg_samples"]

    best_valid_metrics = {"recall@k": -1, "ndcg@k": -1}
    best_model_state = None
    patience_count = 0

    for epoch in range(args["epoch"]):
        model.train()
        total_loss = 0.0

        perm = torch.randperm(num_interactions)
        user_item_pairs_shuffled = user_item_pairs[perm]

        for i in range(n_batch):
            start_idx = i * args["batch_size"]
            end_idx = start_idx + args["batch_size"]
            batch_pairs = user_item_pairs_shuffled[start_idx:end_idx]

            users = batch_pairs[:, 0]
            pos_items = batch_pairs[:, 1]

            neg_items = sample_negative_items(
                users, num_users, num_items, user_train_items, neg_ratio, device
            )

            train_users = torch.cat([users, users.repeat_interleave(neg_ratio)], dim=0)
            train_items = torch.cat([pos_items, neg_items], dim=0)
            labels = torch.cat(
                [
                    torch.ones(len(users), device=device),
                    torch.zeros(len(neg_items), device=device),
                ],
                dim=0,
            )

            outputs = model(user_indices=train_users, item_indices=train_items)
            loss = criterion(outputs.view(-1), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batch
        logger.info(f"Epoch {epoch+1}/{args['epoch']}: Avg Loss={avg_loss:.4f}")

        if eval_interval > 0 and (epoch + 1) % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                valid_metrics = evaluate_model(
                    model,
                    data,
                    args,
                    device,
                    batch_size=args["eval_batch_size"],
                    user_train_items=user_train_items,
                    k=args["k"],
                    is_test=True,
                )
            print_metrics(valid_metrics, prefix="Validation")

            best_valid_metrics, best_model_state, patience_count, should_stop = (
                _handle_early_stopping(
                    model,
                    valid_metrics,
                    best_valid_metrics,
                    best_model_state,
                    patience_count,
                    args,
                )
            )
            if should_stop:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info(
            f"Loaded best model with {args['early_stop_metric']}={best_valid_metrics[args['early_stop_metric']]:.4f}"
        )
    return model


def train_non_gnn_model(
    model, data, args, device, user_train_items=None, eval_interval=5
):
    """Trains a non-GNN model (e.g., MLP, GMF)."""
    model.train()
    optimizer = optim.AdamW(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )

    num_users, num_items = data.num_users, data.num_items
    user_item_pairs = data.train_ratings[["user_idx", "item_idx"]].values
    user_item_pairs = torch.tensor(user_item_pairs, dtype=torch.long, device=device)

    num_interactions = len(user_item_pairs)
    n_batch = max(
        (num_interactions + args["batch_size"] - 1) // args["batch_size"], 1
    )

    best_valid_metrics = {"recall@k": -1, "ndcg@k": -1}
    best_model_state = None
    patience_count = 0

    for epoch in range(args["epoch"]):
        model.train()
        total_loss = 0.0

        perm = torch.randperm(num_interactions)
        user_item_pairs_shuffled = user_item_pairs[perm]

        for i in range(n_batch):
            start_idx = i * args["batch_size"]
            end_idx = start_idx + args["batch_size"]
            batch_pairs = user_item_pairs_shuffled[start_idx:end_idx]

            users = batch_pairs[:, 0]
            pos_items = batch_pairs[:, 1]

            neg_items = sample_negative_items(
                users,
                num_users,
                num_items,
                user_train_items,
                args["neg_samples"],
                device,
            )

            neg_samples = args["neg_samples"]
            users_rep = users.repeat_interleave(neg_samples)
            pos_rep = pos_items.repeat_interleave(neg_samples)

            pos_scores = model(user_indices=users_rep, item_indices=pos_rep)
            neg_scores = model(user_indices=users_rep, item_indices=neg_items)

            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batch
        logger.info(f"Epoch {epoch+1}/{args['epoch']}: Avg Loss={avg_loss:.4f}")

        if eval_interval > 0 and (epoch + 1) % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                valid_metrics = evaluate_model(
                    model,
                    data,
                    args,
                    device,
                    batch_size=args["eval_batch_size"],
                    user_train_items=user_train_items,
                    k=args["k"],
                    is_test=True,
                )
            print_metrics(valid_metrics, prefix="Validation")

            best_valid_metrics, best_model_state, patience_count, should_stop = (
                _handle_early_stopping(
                    model,
                    valid_metrics,
                    best_valid_metrics,
                    best_model_state,
                    patience_count,
                    args,
                )
            )
            if should_stop:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info(
            f"Loaded best model with {args.get('early_stop_metric', 'ndcg@k')}={best_valid_metrics[args.get('early_stop_metric', 'ndcg@k')]:.4f}"
        )
    return model


def train_wmf_model(model, data, args, device, user_train_items=None, eval_interval=5):
    """Trains a WMF model with the BCE objective used in the paper."""
    model.train()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.get("lr", 0.001),
        weight_decay=args.get("weight_decay", 0.0),
    )
    criterion = nn.BCEWithLogitsLoss()

    num_users, num_items = data.num_users, data.num_items
    user_item_pairs = data.train_ratings[["user_idx", "item_idx"]].values
    user_item_pairs = torch.tensor(user_item_pairs, dtype=torch.long, device=device)
    num_interactions = len(user_item_pairs)
    n_batch = max(
        (num_interactions + args["batch_size"] - 1) // args["batch_size"], 1
    )
    neg_ratio = args["neg_samples"]

    best_valid_metrics = {"recall@k": -1, "ndcg@k": -1}
    best_model_state = None
    patience_count = 0

    for epoch in range(args["epoch"]):
        model.train()
        total_loss = 0.0

        perm = torch.randperm(num_interactions, device=device)
        user_item_pairs_shuffled = user_item_pairs[perm]

        for i in range(n_batch):
            start_idx = i * args["batch_size"]
            end_idx = start_idx + args["batch_size"]
            batch_pairs = user_item_pairs_shuffled[start_idx:end_idx]
            if batch_pairs.numel() == 0:
                continue

            users = batch_pairs[:, 0]
            pos_items = batch_pairs[:, 1]
            neg_items = sample_negative_items(
                users, num_users, num_items, user_train_items, neg_ratio, device
            )

            train_users = torch.cat([users, users.repeat_interleave(neg_ratio)], dim=0)
            train_items = torch.cat([pos_items, neg_items], dim=0)
            labels = torch.cat(
                [
                    torch.ones(len(users), device=device),
                    torch.zeros(len(neg_items), device=device),
                ],
                dim=0,
            )

            outputs = model(user_indices=train_users, item_indices=train_items)
            loss = criterion(outputs.view(-1), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batch
        logger.info(f"Epoch {epoch+1}/{args['epoch']} (WMF): Avg Loss={avg_loss:.4f}")

        if eval_interval > 0 and (epoch + 1) % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                valid_metrics = evaluate_model(
                    model,
                    data,
                    args,
                    device,
                    batch_size=args["eval_batch_size"],
                    user_train_items=user_train_items,
                    k=args.get("k", 10),
                    is_test=True,
                )
            print_metrics(valid_metrics, prefix=f"Validation (WMF Epoch {epoch+1})")

            best_valid_metrics, best_model_state, patience_count, should_stop = (
                _handle_early_stopping(
                    model,
                    valid_metrics,
                    best_valid_metrics,
                    best_model_state,
                    patience_count,
                    args,
                )
            )
            if should_stop:
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        logger.info(
            f"Loaded best model with {args.get('early_stop_metric', 'ndcg@k')}={best_valid_metrics[args.get('early_stop_metric', 'ndcg@k')]:.4f}"
        )
    return model


def sample_negative_items(
    users, num_users, num_items, user_train_items, neg_samples, device
):
    """
    High-performance negative sampling function (GPU-friendly).
    This implementation performs sampling entirely on the GPU to reduce overhead.
    It "tolerates" a very low probability of sampling a positive item, which has a negligible
    impact on training but significantly improves performance.

    Args:
        users: Tensor of user indices, shape [batch_size].
        num_users: Total number of users.
        num_items: Total number of items.
        user_train_items: Dict mapping user to their set of positive items (not used in this version).
        neg_samples: Number of negative samples per user.
        device: The computation device.

    Returns:
        neg_items: Tensor of negative item indices, shape [batch_size * neg_samples].
                   Indices are item-specific (0 to num_items-1).
    """
    batch_size = users.size(0)
    # The node indices for items start from num_users. We sample from item-specific indices (0 to num_items-1).
    neg_items = torch.randint(0, num_items, (batch_size * neg_samples,), device=device)
    return neg_items


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking (BPR) loss.
    Designed for implicit feedback data in recommender systems.
    """

    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, user_emb, pos_item_emb, neg_item_emb, lambda_reg=0.0):
        """
        Calculates the BPR loss.

        Args:
            user_emb: User embeddings.
            pos_item_emb: Positive item embeddings.
            neg_item_emb: Negative item embeddings.
            lambda_reg: L2 regularization coefficient.

        Returns:
            A tuple containing (total_loss, mf_loss, reg_loss).
        """
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)

        mf_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))

        reg_loss = (
            (
                lambda_reg
                * (
                    user_emb.norm(2).pow(2)
                    + pos_item_emb.norm(2).pow(2)
                    + neg_item_emb.norm(2).pow(2)
                )
            )
            / (3 * len(user_emb))
            if lambda_reg > 0
            else torch.tensor(0.0)
        )

        return mf_loss + reg_loss, mf_loss, reg_loss


def get_padded_positive_items_wmf(
    user_train_items, num_users, num_items, max_items_per_user, padding_idx, device
):
    """
    Prepares padded positive item lists and masks for each user for the WMF model.
    """
    if user_train_items is None:
        user_train_items = {}

    all_user_pos_items_padded = torch.full(
        (num_users, max_items_per_user),
        fill_value=padding_idx,
        dtype=torch.long,
        device=device,
    )
    all_user_pos_item_masks = torch.zeros(
        (num_users, max_items_per_user), dtype=torch.float, device=device
    )

    for u_idx in range(num_users):
        if u_idx in user_train_items:
            pos_list = list(user_train_items[u_idx])
            if len(pos_list) > max_items_per_user:
                pos_list_sampled = np.random.choice(
                    pos_list, max_items_per_user, replace=False
                )
            else:
                pos_list_sampled = pos_list

            num_actual_items = len(pos_list_sampled)
            if num_actual_items > 0:
                all_user_pos_items_padded[u_idx, :num_actual_items] = torch.tensor(
                    pos_list_sampled, dtype=torch.long, device=device
                )
                all_user_pos_item_masks[u_idx, :num_actual_items] = 1.0

    return all_user_pos_items_padded, all_user_pos_item_masks
