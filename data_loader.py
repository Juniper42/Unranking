import gc
import multiprocessing as mp
import os
import pickle
import random
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from logger import Logger
from utils import determine_is_gnn

logger = Logger.get_logger("DataLoader")


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_cache_path(dataset_name, test_size, val_size=0.1, backbone="lightgcn"):
    """
    Gets the cache file path, distinguishing cache directories and paths based on model type.
    """
    is_gnn = determine_is_gnn(backbone)
    model_type = "gnn" if is_gnn else "non_gnn"

    if not (0 <= test_size <= 1 and 0 <= val_size <= 1 and (test_size + val_size) <= 1):
        raise ValueError(
            f"test_size ({test_size}) and val_size ({val_size}) must be in [0, 1] and their sum cannot exceed 1."
        )
    train_share = 1.0 - test_size - val_size

    cache_dir_name = (
        f"train{train_share:.2f}_val{val_size:.2f}_test{test_size:.2f}_{model_type}"
    )
    cache_dir_path = os.path.join(f"./data/{dataset_name}/cache/", cache_dir_name)

    data_pkl_path = os.path.join(cache_dir_path, "data.pkl")
    info_txt_path = os.path.join(cache_dir_path, "info.txt")

    return {
        "cache_dir_path": cache_dir_path,
        "data_pkl_path": data_pkl_path,
        "info_txt_path": info_txt_path,
        "is_gnn": is_gnn,
        "train_share_for_path": train_share,
        "model_type": model_type,
    }


def _process_user_split(user_group, test_size, val_size):
    """Processes a single user's data for splitting."""
    user_id, user_data = user_group
    total = len(user_data)
    original_columns = user_data.columns

    if total <= 1:
        return (
            user_id,
            user_data.copy(),
            pd.DataFrame(columns=original_columns),
            pd.DataFrame(columns=original_columns),
        )

    shuffled_data = user_data.sample(frac=1, random_state=42).reset_index(drop=True)

    num_test = int(total * test_size)
    num_val = int(total * val_size)

    if num_test + num_val >= total:
        num_test = total - num_val if total > num_val else 0
        if num_test + num_val > total:
            num_val = total - num_test

    num_train = total - num_test - num_val

    train_df = shuffled_data.iloc[:num_train]
    val_df = shuffled_data.iloc[num_train : num_train + num_val]
    test_df = shuffled_data.iloc[num_train + num_val :]

    return user_id, train_df, val_df, test_df


def user_random_split(ratings_df, test_size=0.2, val_size=0.1, num_workers=4):
    """
    Splits the dataset into training, validation, and test sets randomly per user.
    """
    logger.info("Splitting dataset randomly by user...")
    user_groups = list(ratings_df.groupby("user_id"))

    if num_workers > 1 and len(ratings_df) > 10000:
        import math
        from concurrent.futures import ProcessPoolExecutor

        chunksize = max(1, math.ceil(len(user_groups) / (num_workers * 4)))
        process_fn = partial(
            _process_user_split, test_size=test_size, val_size=val_size
        )

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_fn, user_groups, chunksize=chunksize),
                    total=len(user_groups),
                    desc="Processing user data",
                )
            )
    else:
        results = [
            _process_user_split(user_group, test_size, val_size)
            for user_group in tqdm(user_groups, desc="Processing user data")
        ]

    train_dfs, val_dfs, test_dfs = zip(*[(r[1], r[2], r[3]) for r in results])

    train_ratings = pd.concat(train_dfs, ignore_index=True)
    val_ratings = (
        pd.concat(val_dfs, ignore_index=True)
        if val_size > 0
        else pd.DataFrame(columns=ratings_df.columns)
    )
    test_ratings = pd.concat(test_dfs, ignore_index=True)

    logger.info(
        f"Split results: {len(train_ratings)} train, {len(val_ratings)} validation, {len(test_ratings)} test interactions."
    )

    return train_ratings, val_ratings, test_ratings


def _apply_10_core_filter(ratings):
    """Iteratively applies 10-core filtering to users and items."""
    while True:
        user_counts = ratings["user_id"].value_counts()
        item_counts = ratings["item_id"].value_counts()

        initial_rows = len(ratings)

        users_to_keep = user_counts[user_counts >= 10].index
        ratings = ratings[ratings["user_id"].isin(users_to_keep)]

        items_to_keep = item_counts[item_counts >= 10].index
        ratings = ratings[ratings["item_id"].isin(items_to_keep)]

        if len(ratings) == initial_rows:
            break

    return ratings


def load_dataset_raw(dataset_name, data_dir=None):
    """Loads a raw dataset from its source file."""
    if data_dir is None:
        data_dir = f"./data/{dataset_name}"

    if dataset_name == "ml-1m":
        rating_file = "ratings.dat"
        ratings = pd.read_csv(
            os.path.join(data_dir, rating_file),
            sep="::",
            engine="python",
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
        )
        ratings = ratings[ratings["rating"] >= 3]
        filtered_ratings = _apply_10_core_filter(ratings)

    elif dataset_name in ["yelp2018", "amazon-book"]:
        rating_file = "total.txt"
        user_items = []
        with open(os.path.join(data_dir, rating_file), "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) > 1:
                    user_id = parts[0]
                    for item_id in parts[1:]:
                        user_items.append({"user_id": user_id, "item_id": item_id})
        ratings = pd.DataFrame(user_items)
        ratings["rating"] = 1
        ratings["timestamp"] = range(len(ratings))
        filtered_ratings = _apply_10_core_filter(ratings)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    logger.info("Performing data quality checks...")
    if (
        len(
            set(filtered_ratings["user_id"].unique())
            & set(filtered_ratings["item_id"].unique())
        )
        > 0
    ):
        logger.warning(
            "User and item IDs overlap, which might cause issues in graph construction."
        )

    if filtered_ratings.isnull().sum().sum() > 0:
        logger.warning("Dropping rows with null values.")
        filtered_ratings.dropna(inplace=True)

    logger.info(
        f"Dataset loaded: {filtered_ratings['user_id'].nunique()} users, {filtered_ratings['item_id'].nunique()} items, "
        f"{len(filtered_ratings)} interactions. Avg interactions/user: {len(filtered_ratings)/filtered_ratings['user_id'].nunique():.2f}"
    )

    return filtered_ratings


def _offset_item_nodes(edge_index, num_users):
    """Converts forward user-item edges to global bipartite node ids."""
    if edge_index.numel() == 0:
        return edge_index
    edge_index = edge_index.clone()
    edge_index[1] = edge_index[1] + num_users
    return edge_index


def _ensure_global_item_nodes(data, is_gnn=True):
    """
    Migrates cached data created by older code where item ids were not offset.
    The rest of the codebase assumes item nodes live in
    [num_users, num_users + num_items).
    """
    if not hasattr(data, "train_edge_index_original"):
        return data
    if data.train_edge_index_original.numel() == 0:
        return data
    if int(data.train_edge_index_original[1].min().item()) >= data.num_users:
        return data

    logger.warning(
        "Cached edge_index uses local item ids; migrating to global item node ids."
    )

    for split in ("train", "val", "test"):
        attr = f"{split}_edge_index_original"
        if hasattr(data, attr):
            setattr(data, attr, _offset_item_nodes(getattr(data, attr), data.num_users))

    if is_gnn:
        data.train_edge_index = torch.cat(
            [data.train_edge_index_original, data.train_edge_index_original.flip(0)],
            dim=1,
        )
        data.val_edge_index = torch.cat(
            [data.val_edge_index_original, data.val_edge_index_original.flip(0)],
            dim=1,
        )
        data.test_edge_index = torch.cat(
            [data.test_edge_index_original, data.test_edge_index_original.flip(0)],
            dim=1,
        )
    else:
        data.train_edge_index = data.train_edge_index_original
        data.val_edge_index = data.val_edge_index_original
        data.test_edge_index = data.test_edge_index_original

    data.edge_index = data.train_edge_index
    data.train_edge_index_after_remove = data.train_edge_index.clone()
    return data


def build_pyg_data_from_df(ratings_df, user_id_map, item_id_map, device, is_gnn=True):
    """Builds a PyTorch Geometric Data object from a ratings DataFrame."""
    data = Data()
    data.num_users = len(user_id_map)
    data.num_items = len(item_id_map)

    # Separate dataframes by split
    train_df = ratings_df[ratings_df["split"] == "train"]
    val_df = ratings_df[ratings_df["split"] == "val"]
    test_df = ratings_df[ratings_df["split"] == "test"]

    def create_edge_index(df):
        if df.empty:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        local_edge_index = torch.tensor(
            [df["user_idx"].values, df["item_idx"].values],
            dtype=torch.long,
            device=device,
        )
        return _offset_item_nodes(local_edge_index, data.num_users)

    train_edges = create_edge_index(train_df)
    val_edges = create_edge_index(val_df)
    test_edges = create_edge_index(test_df)

    if is_gnn:
        data.train_edge_index_original = train_edges
        data.val_edge_index_original = val_edges
        data.test_edge_index_original = test_edges

        data.train_edge_index = torch.cat([train_edges, train_edges.flip(0)], dim=1)
        data.val_edge_index = torch.cat([val_edges, val_edges.flip(0)], dim=1)
        data.test_edge_index = torch.cat([test_edges, test_edges.flip(0)], dim=1)

        # The full graph for GNNs is typically the training graph
        data.edge_index = data.train_edge_index
    else:
        # For non-GNN models, edges are directional and typically only from training set
        data.train_edge_index = train_edges
        data.val_edge_index = val_edges
        data.test_edge_index = test_edges

        # Store original as well for consistency
        data.train_edge_index_original = train_edges
        data.val_edge_index_original = val_edges
        data.test_edge_index_original = test_edges
        data.edge_index = data.train_edge_index

    # Initialize fields for unlearning
    data.train_edge_index_after_remove = data.train_edge_index.clone()

    return data


def load_dataset(dataset_name, device, backbone="lightgcn", **kwargs):
    """Loads and preprocesses a dataset, using a caching mechanism."""
    test_size = kwargs.get("test_size", 0.1)
    val_size = kwargs.get("val_size", 0.1)
    use_cache = kwargs.get("use_cache", True)

    cache_info = get_cache_path(dataset_name, test_size, val_size, backbone)
    cache_dir_path = cache_info["cache_dir_path"]
    data_pkl_path = cache_info["data_pkl_path"]
    info_txt_path = cache_info["info_txt_path"]
    is_gnn_model = cache_info["is_gnn"]

    if use_cache and os.path.exists(data_pkl_path):
        logger.info(f"Loading cached data from {data_pkl_path}")
        with open(data_pkl_path, "rb") as f:
            data = pickle.load(f)
        with open(info_txt_path, "r") as f:
            logger.info(f"Cached dataset info:\n{f.read()}")

        # The 'ratings' df might be needed for some unlearning methods, ensure it exists
        if not hasattr(data, "ratings"):
            data.ratings = pd.concat(
                [data.train_ratings, data.val_ratings, data.test_ratings],
                ignore_index=True,
            )

        data = _ensure_global_item_nodes(data, is_gnn=is_gnn_model)
        return data, data.ratings

    logger.info(
        f"No cache found or cache is disabled. Processing raw dataset: {dataset_name} for {cache_info['model_type']} model."
    )

    raw_ratings_df = load_dataset_raw(dataset_name)
    train_df, val_df, test_df = user_random_split(
        raw_ratings_df, test_size, val_size, num_workers=max(1, mp.cpu_count() // 2)
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    all_ratings = pd.concat([train_df, val_df, test_df], ignore_index=True)

    all_user_ids = sorted(raw_ratings_df["user_id"].unique())
    all_item_ids = sorted(raw_ratings_df["item_id"].unique())
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(all_user_ids)}
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(all_item_ids)}

    all_ratings["user_idx"] = all_ratings["user_id"].map(user_id_map)
    all_ratings["item_idx"] = all_ratings["item_id"].map(item_id_map)
    all_ratings.dropna(subset=["user_idx", "item_idx"], inplace=True)
    all_ratings["user_idx"] = all_ratings["user_idx"].astype(int)
    all_ratings["item_idx"] = all_ratings["item_idx"].astype(int)

    data = build_pyg_data_from_df(
        all_ratings, user_id_map, item_id_map, device, is_gnn=is_gnn_model
    )

    data.user_id_map = user_id_map
    data.item_id_map = item_id_map
    data.train_ratings = all_ratings[all_ratings["split"] == "train"].copy()
    data.val_ratings = all_ratings[all_ratings["split"] == "val"].copy()
    data.test_ratings = all_ratings[all_ratings["split"] == "test"].copy()
    data.ratings = all_ratings

    num_train = data.train_edge_index_original.size(1)
    num_val = data.val_edge_index_original.size(1)
    num_test = data.test_edge_index_original.size(1)

    info_text = f"""Dataset: {dataset_name} ({cache_info['model_type']} model)
Parameters: test_size={test_size:.2f}, val_size={val_size:.2f}
Statistics:
  Users: {data.num_users}, Items: {data.num_items}
  Total Interactions: {len(all_ratings)}
  Train Interactions: {num_train}
  Validation Interactions: {num_val}
  Test Interactions: {num_test}
"""
    logger.info("Generated Dataset Information:")
    logger.info(info_text)

    if use_cache:
        os.makedirs(cache_dir_path, exist_ok=True)
        logger.info(f"Caching processed data to: {data_pkl_path}")
        with open(data_pkl_path, "wb") as f:
            pickle.dump(data, f)
        with open(info_txt_path, "w") as f:
            f.write(info_text)

    return data, all_ratings
