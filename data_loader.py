import json
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

# Paper: "A 10-core filtering procedure was applied to ensure data quality".
DEFAULT_CORE = 10
DEFAULT_MIN_RATING = 4.0


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


def get_cache_path(
    dataset_name,
    test_size,
    val_size=0.1,
    backbone="lightgcn",
    split_method="temporal",
    min_rating=DEFAULT_MIN_RATING,
    core=DEFAULT_CORE,
):
    """Get the cache path for a dataset and preprocessing configuration.

    The cache key includes the split protocol, the rating threshold, and the
    k-core threshold so that a cache produced by another preprocessing
    configuration is never reused silently.
    """
    is_gnn = determine_is_gnn(backbone)
    model_type = "gnn" if is_gnn else "non_gnn"

    if not (0 <= test_size <= 1 and 0 <= val_size <= 1 and (test_size + val_size) <= 1):
        raise ValueError(
            f"test_size ({test_size}) and val_size ({val_size}) must be in [0, 1] "
            "and their sum cannot exceed 1."
        )
    train_share = 1.0 - test_size - val_size

    cache_dir_name = (
        f"train{train_share:.2f}_val{val_size:.2f}_test{test_size:.2f}"
        f"_{model_type}_{split_method}_r{float(min_rating):g}_c{int(core)}"
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


# ----------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------
def _process_user_split(user_group, test_size, val_size):
    """Processes a single user's data for random splitting."""
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
    """Randomly split each user's interactions (legacy option).

    The paper uses a chronological split; this function is retained only for
    ``--split_method random``.
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
        f"Split results: {len(train_ratings)} train, {len(val_ratings)} validation, "
        f"{len(test_ratings)} test interactions."
    )
    return train_ratings, val_ratings, test_ratings


def user_temporal_split(ratings_df, test_size=0.1, val_size=0.1):
    """Chronological 8:1:1 split used in the paper.

    For each user, interactions are sorted by timestamp; the earliest
    ``1 - test_size - val_size`` fraction forms the training pool, the next
    ``val_size`` fraction the validation set, and the most recent
    ``test_size`` fraction the test set.  This ordering is what makes the
    test set reflect the user's current preference.
    """
    if "timestamp" not in ratings_df.columns:
        raise ValueError(
            "A chronological split requires a 'timestamp' column. "
            "Use --split_method random only for exploratory runs."
        )

    logger.info("Splitting dataset chronologically by user (8:1:1 by default)...")
    ordered = ratings_df.sort_values(
        ["user_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    ordered["_position"] = ordered.groupby("user_id", sort=False).cumcount()
    ordered["_count"] = ordered.groupby("user_id", sort=False)["_position"].transform(
        "max"
    ) + 1

    # Round the requested fractions to whole interactions, keeping at least
    # one training interaction for users with enough data.
    test_count = np.floor(ordered["_count"] * float(test_size) + 1e-9).astype(int)
    val_count = np.floor(ordered["_count"] * float(val_size) + 1e-9).astype(int)
    # 10-core filtering guarantees at least 10 interactions per user, so the
    # rounded counts yield non-empty validation and test sets by default.
    remaining = ordered["_count"] - test_count - val_count
    adjust = remaining < 1
    if adjust.any():
        overflow = (1 - remaining).clip(lower=0)
        test_count = (test_count - overflow).clip(lower=0)
        val_count = (val_count - overflow).clip(lower=0)
        remaining = ordered["_count"] - test_count - val_count

    position = ordered["_position"]
    is_train = position < remaining
    is_val = (position >= remaining) & (position < remaining + val_count)
    is_test = position >= remaining + val_count

    train_df = ordered[is_train].copy()
    val_df = ordered[is_val].copy() if val_size > 0 else ordered.iloc[0:0].copy()
    test_df = ordered[is_test].copy()

    train_df = train_df.drop(columns=["_position", "_count"])
    val_df = val_df.drop(columns=["_position", "_count"])
    test_df = test_df.drop(columns=["_position", "_count"])

    logger.info(
        f"Chronological split results: {len(train_df)} train, "
        f"{len(val_df)} validation, {len(test_df)} test interactions."
    )
    return train_df, val_df, test_df


# ----------------------------------------------------------------------
# Raw dataset loading
# ----------------------------------------------------------------------
def _apply_k_core_filter(ratings, core=DEFAULT_CORE):
    """Iteratively applies k-core filtering to users and items."""
    core = max(int(core), 1)
    while True:
        user_counts = ratings["user_id"].value_counts()
        item_counts = ratings["item_id"].value_counts()

        initial_rows = len(ratings)

        users_to_keep = user_counts[user_counts >= core].index
        ratings = ratings[ratings["user_id"].isin(users_to_keep)]

        items_to_keep = item_counts[item_counts >= core].index
        ratings = ratings[ratings["item_id"].isin(items_to_keep)]

        if len(ratings) == initial_rows:
            break

    return ratings


def _coerce_timestamp(series):
    """Convert numeric or date-like timestamps to integer seconds.

    Missing values (NaN/NaT) are preserved as NaN so that
    ``timestamps_available`` can detect partial timestamp coverage.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().all():
        return parsed.astype("int64") // 10**9
    converted = pd.Series(np.nan, index=series.index, dtype="float64")
    mask = parsed.notna()
    converted.loc[mask] = (parsed.loc[mask].astype("int64") // 10**9).astype(
        "float64"
    )
    return converted


def _read_table(path):
    """Read a ratings table and normalize its column names."""
    if path.endswith(".tsv"):
        frame = pd.read_csv(path, sep="\t")
    else:
        frame = pd.read_csv(path)

    aliases = {
        "user_id": {"user_id", "user", "uid", "userid", "reviewerid", "reviewer_id"},
        "item_id": {
            "item_id",
            "item",
            "iid",
            "itemid",
            "asin",
            "business_id",
            "movie_id",
            "movieid",
            "mid",
        },
        "rating": {"rating", "stars", "overall", "score"},
        "timestamp": {"timestamp", "time", "date", "unixreviewtime", "unix_review_time"},
    }
    rename = {}
    for column in frame.columns:
        key = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for canonical, names in aliases.items():
            if key in names and canonical not in rename.values():
                rename[column] = canonical
                break
    frame = frame.rename(columns=rename)

    if "user_id" not in frame.columns or "item_id" not in frame.columns:
        raise ValueError(f"{path} must contain user and item columns.")
    if "rating" not in frame.columns:
        frame["rating"] = 1.0
    if "timestamp" not in frame.columns:
        frame["timestamp"] = np.nan
    else:
        frame["timestamp"] = _coerce_timestamp(frame["timestamp"])

    return frame[["user_id", "item_id", "rating", "timestamp"]].copy()


def _load_ml_ratings(data_dir, min_rating):
    dat_path = os.path.join(data_dir, "ratings.dat")
    if os.path.exists(dat_path):
        ratings = pd.read_csv(
            dat_path,
            sep="::",
            engine="python",
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
        )
    else:
        csv_path = os.path.join(data_dir, "ratings.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Expected ratings.dat or ratings.csv in {data_dir}."
            )
        ratings = _read_table(csv_path)

    ratings = ratings[ratings["rating"] >= float(min_rating)].copy()
    return ratings, False


def _load_netflix_ratings(data_dir, min_rating):
    csv_path = os.path.join(data_dir, "ratings.csv")
    if os.path.exists(csv_path):
        ratings = _read_table(csv_path)
        ratings = ratings[ratings["rating"] >= float(min_rating)].copy()
        return ratings, False

    import glob

    combined_files = sorted(glob.glob(os.path.join(data_dir, "combined_data_*.txt")))
    if not combined_files:
        raise FileNotFoundError(
            f"Expected ratings.csv or combined_data_*.txt in {data_dir}."
        )

    users, items, scores, times = [], [], [], []
    for path in combined_files:
        logger.info(f"Reading Netflix file {path} ...")
        current_item = None
        with open(path, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.endswith(":"):
                    current_item = line[:-1]
                    continue
                parts = line.split(",")
                if len(parts) != 3 or current_item is None:
                    continue
                user, rating, date = parts
                if float(rating) < float(min_rating):
                    continue
                users.append(user)
                items.append(current_item)
                scores.append(float(rating))
                times.append(date)
    ratings = pd.DataFrame(
        {
            "user_id": users,
            "item_id": items,
            "rating": scores,
            "timestamp": times,
        }
    )
    ratings["timestamp"] = _coerce_timestamp(ratings["timestamp"])
    return ratings, False


def _load_total_txt(data_dir):
    path = os.path.join(data_dir, "total.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected ratings.csv/ratings.tsv or total.txt in {data_dir}."
        )
    user_items = []
    with open(path, "r") as handle:
        for line in handle:
            parts = line.strip().split(" ")
            if len(parts) > 1:
                user_id = parts[0]
                for item_id in parts[1:]:
                    user_items.append((user_id, item_id))
    ratings = pd.DataFrame(user_items, columns=["user_id", "item_id"])
    ratings["rating"] = 1.0
    ratings["timestamp"] = np.nan
    return ratings


def _attach_timestamps(ratings, data_dir, dataset_name):
    """Recover interaction timestamps for preprocessed Yelp/Amazon files.

    ``total.txt`` does not carry timestamps.  The preprocessing used for the
    paper recovers them from the raw review collections.  This function
    supports the resulting ``timestamps.csv``/``timestamps.tsv`` file, and
    additionally joins Yelp ``review.json`` or Amazon ``reviews*.csv`` files
    when their identifiers match the preprocessed ids.
    """
    timestamp_files = [
        os.path.join(data_dir, "timestamps.csv"),
        os.path.join(data_dir, "timestamps.tsv"),
    ]
    for path in timestamp_files:
        if os.path.exists(path):
            table = _read_table(path)
            merged = ratings.drop(columns=["timestamp"]).merge(
                table[["user_id", "item_id", "timestamp"]],
                on=["user_id", "item_id"],
                how="left",
            )
            logger.info(f"Recovered timestamps from {path}.")
            return merged

    yelp_review = os.path.join(data_dir, "review.json")
    if dataset_name == "yelp2018" and os.path.exists(yelp_review):
        logger.info("Recovering Yelp timestamps from review.json ...")
        lookup = {}
        with open(yelp_review, "r") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lookup[(str(row.get("user_id")), str(row.get("business_id")))] = (
                    row.get("date")
                )
        if lookup:
            keys = list(zip(ratings["user_id"].astype(str), ratings["item_id"].astype(str)))
            ratings["timestamp"] = [lookup.get(key) for key in keys]
            ratings["timestamp"] = _coerce_timestamp(ratings["timestamp"])

    if dataset_name == "amazon-book":
        import glob

        review_files = sorted(glob.glob(os.path.join(data_dir, "reviews*.csv")))
        if review_files:
            logger.info("Recovering Amazon timestamps from reviews*.csv ...")
            lookup = {}
            for path in review_files:
                try:
                    frame = pd.read_csv(
                        path,
                        usecols=["reviewerID", "asin", "unixReviewTime"],
                    )
                except (ValueError, KeyError):
                    continue
                for user, item, time in zip(
                    frame["reviewerID"].astype(str),
                    frame["asin"].astype(str),
                    frame["unixReviewTime"],
                ):
                    lookup[(user, item)] = time
            if lookup:
                keys = list(
                    zip(ratings["user_id"].astype(str), ratings["item_id"].astype(str))
                )
                ratings["timestamp"] = [lookup.get(key) for key in keys]
                ratings["timestamp"] = _coerce_timestamp(ratings["timestamp"])

    return ratings


def load_dataset_raw(
    dataset_name,
    data_dir=None,
    min_rating=DEFAULT_MIN_RATING,
    core=DEFAULT_CORE,
):
    """Load and preprocess one of the five paper datasets.

    Returns ``(ratings, meta)``.  ``meta["timestamps_available"]`` records
    whether all retained interactions have a real timestamp; if not, a row
    order surrogate is used and a warning is emitted.
    """
    if data_dir is None:
        data_dir = f"./data/{dataset_name}"

    implicit_data = False
    if dataset_name in {"ml-1m", "ml-10m"}:
        ratings, implicit_data = _load_ml_ratings(data_dir, min_rating)
    elif dataset_name == "netflix":
        ratings, implicit_data = _load_netflix_ratings(data_dir, min_rating)
    elif dataset_name in {"yelp2018", "amazon-book"}:
        csv_path = None
        for candidate in ("ratings.csv", "ratings.tsv"):
            path = os.path.join(data_dir, candidate)
            if os.path.exists(path):
                csv_path = path
                break
        if csv_path is not None:
            ratings = _read_table(csv_path)
            ratings = ratings[ratings["rating"] >= float(min_rating)].copy()
        else:
            ratings = _load_total_txt(data_dir)
            implicit_data = True
            ratings = _attach_timestamps(ratings, data_dir, dataset_name)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if ratings.empty:
        raise ValueError(f"Dataset {dataset_name} is empty after preprocessing.")

    # Duplicate (user, item) pairs are one interaction in D.  Keep the latest
    # observed rating when timestamps are available.
    if ratings["timestamp"].notna().any():
        ratings = ratings.sort_values("timestamp", kind="mergesort")
        ratings = ratings.drop_duplicates(["user_id", "item_id"], keep="last")
    else:
        ratings = ratings.drop_duplicates(["user_id", "item_id"], keep="last")

    timestamps_available = bool(ratings["timestamp"].notna().all())
    if not timestamps_available:
        logger.warning(
            "No timestamps are available for %s. Falling back to row order as "
            "a surrogate for the chronological split. The paper's Yelp2018 "
            "and Amazon-Book preprocessing recovers timestamps from the raw "
            "review collections; provide timestamps.csv (user_id, item_id, "
            "timestamp) to use the paper protocol.",
            dataset_name,
        )
        ratings["timestamp"] = np.arange(len(ratings), dtype=np.int64)
    else:
        ratings["timestamp"] = ratings["timestamp"].astype("int64")

    ratings = _apply_k_core_filter(ratings, core=core)
    ratings = ratings.reset_index(drop=True)

    logger.info(
        "Dataset loaded: %d users, %d items, %d interactions (min_rating=%s, "
        "%d-core, implicit_file=%s, timestamps=%s).",
        ratings["user_id"].nunique(),
        ratings["item_id"].nunique(),
        len(ratings),
        min_rating,
        core,
        implicit_data,
        timestamps_available,
    )

    return ratings, {
        "timestamps_available": timestamps_available,
        "implicit_file": implicit_data,
        "min_rating": float(min_rating),
        "core": int(core),
    }


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------
def _offset_item_nodes(edge_index, num_users):
    """Converts forward user-item edges to global bipartite node ids."""
    if edge_index.numel() == 0:
        return edge_index
    edge_index = edge_index.clone()
    edge_index[1] = edge_index[1] + num_users
    return edge_index


def _ensure_global_item_nodes(data, is_gnn=True):
    """Migrate cached data created by older code with local item ids."""
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

        data.edge_index = data.train_edge_index
    else:
        data.train_edge_index = train_edges
        data.val_edge_index = val_edges
        data.test_edge_index = test_edges

        data.train_edge_index_original = train_edges
        data.val_edge_index_original = val_edges
        data.test_edge_index_original = test_edges
        data.edge_index = data.train_edge_index

    data.train_edge_index_after_remove = data.train_edge_index.clone()
    return data


def load_dataset(dataset_name, device, backbone="lightgcn", **kwargs):
    """Load and preprocess a dataset, using a caching mechanism."""
    test_size = kwargs.get("test_size", 0.1)
    val_size = kwargs.get("val_size", 0.1)
    use_cache = kwargs.get("use_cache", True)
    split_method = kwargs.get("split_method", "temporal")
    min_rating = kwargs.get("min_rating", DEFAULT_MIN_RATING)
    core = kwargs.get("core", DEFAULT_CORE)

    cache_info = get_cache_path(
        dataset_name,
        test_size,
        val_size,
        backbone,
        split_method=split_method,
        min_rating=min_rating,
        core=core,
    )
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
        if not hasattr(data, "ratings"):
            data.ratings = pd.concat(
                [data.train_ratings, data.val_ratings, data.test_ratings],
                ignore_index=True,
            )
        data = _ensure_global_item_nodes(data, is_gnn=is_gnn_model)
        return data, data.ratings

    logger.info(
        f"No cache found or cache disabled. Processing raw dataset: "
        f"{dataset_name} for {cache_info['model_type']} model."
    )

    raw_ratings_df, meta = load_dataset_raw(
        dataset_name, min_rating=min_rating, core=core
    )
    if split_method == "temporal":
        train_df, val_df, test_df = user_temporal_split(
            raw_ratings_df, test_size, val_size
        )
    elif split_method == "random":
        train_df, val_df, test_df = user_random_split(
            raw_ratings_df, test_size, val_size, num_workers=max(1, mp.cpu_count() // 2)
        )
    else:
        raise ValueError(f"Unsupported split_method: {split_method}")

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
    data.timestamps_available = meta["timestamps_available"]
    data.split_method = split_method
    data.min_rating = float(min_rating)
    data.core = int(core)

    num_train = data.train_edge_index_original.size(1)
    num_val = data.val_edge_index_original.size(1)
    num_test = data.test_edge_index_original.size(1)

    info_text = f"""Dataset: {dataset_name} ({cache_info['model_type']} model)
Parameters: test_size={test_size:.2f}, val_size={val_size:.2f}, \
split={split_method}, min_rating={min_rating:g}, core={core}
Statistics:
  Users: {data.num_users}, Items: {data.num_items}
  Total Interactions: {len(all_ratings)}
  Train Interactions: {num_train}
  Validation Interactions: {num_val}
  Test Interactions: {num_test}
  Timestamps available: {meta['timestamps_available']}
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
