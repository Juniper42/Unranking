import copy
import hashlib
import json
import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from logger import Logger
from trainer import train_model
from utils import get_model

logger = Logger.get_logger("SISA")


class SISA:
    """
    Implementation of SISA (Sharded, Isolated, Sliced, and Aggregated training).
    This unlearning method is based on data sharding:
    1. Sharding: The training data is split into multiple disjoint shards.
    2. Isolation: An independent sub-model is trained on each shard.
    3. Slicing: (Optional) Each shard can be trained incrementally in slices.
    4. Aggregation: The predictions from all sub-models are aggregated to form the final output.
    When an unlearning request arrives, only the sub-models whose shards contain the
    data to be forgotten need to be retrained.
    """

    def __init__(self, original_model, data, args, device):
        self.original_model = original_model
        self.data = data
        self.args = args
        self.device = device
        self.num_shards = args.get("sisa_num_shards", 20)
        self.aggregation_strategy = args.get("sisa_aggregation", "uniform")
        self.shard_models: List[nn.Module] = []
        self.shard_data: List[torch.Tensor] = []
        self.shard_weights: List[float] = []

        self.cache_dir = f"./sisa_cache/{args['dataset']}_{args['backbone']}"
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(
            f"Initialized SISA with {self.num_shards} shards. Aggregation: {self.aggregation_strategy}."
        )

    def _create_data_shards(self) -> List[torch.Tensor]:
        """Splits the training data into a specified number of shards."""
        logger.info("Sharding training data...")
        train_edges = self.data.train_edge_index
        num_edges = train_edges.size(1)
        indices = torch.randperm(num_edges)

        shards = []
        shard_size = num_edges // self.num_shards
        for i in range(self.num_shards):
            start = i * shard_size
            end = (i + 1) * shard_size if i < self.num_shards - 1 else num_edges
            shard_indices = indices[start:end]
            shards.append(train_edges[:, shard_indices])

        logger.info(
            f"Data sharding complete. Shard sizes: {[s.size(1) for s in shards]}"
        )
        return shards

    def _train_shard_model(self, shard_id: int, shard_edges: torch.Tensor) -> nn.Module:
        """Trains a sub-model on a single data shard, using caching if possible."""
        shard_hash = hashlib.md5(
            str(shard_edges.cpu().numpy().tolist()).encode()
        ).hexdigest()
        cache_path = os.path.join(self.cache_dir, f"shard_{shard_id}_{shard_hash}.pt")

        shard_model = get_model(self.args, self.data.num_users, self.data.num_items).to(
            self.device
        )

        if os.path.exists(cache_path):
            logger.info(f"Loading cached model for shard {shard_id}.")
            shard_model.load_state_dict(
                torch.load(cache_path, map_location=self.device)
            )
            return shard_model

        logger.info(
            f"Training model for shard {shard_id} with {shard_edges.size(1)} edges..."
        )
        shard_data = copy.deepcopy(self.data)
        shard_data.train_edge_index = shard_edges

        trained_model = train_model(shard_model, shard_data, self.args, self.device)

        torch.save(trained_model.state_dict(), cache_path)
        logger.info(f"Shard {shard_id} model trained and cached.")
        return trained_model

    def _train_all_shards(self):
        """Creates data shards and trains all corresponding sub-models."""
        logger.info("Training all shard models for the first time...")
        self.shard_data = self._create_data_shards()
        self.shard_models = [
            self._train_shard_model(i, edges) for i, edges in enumerate(self.shard_data)
        ]

        total_edges = self.data.train_edge_index.size(1)
        if self.aggregation_strategy == "weighted":
            self.shard_weights = [
                shard.size(1) / total_edges for shard in self.shard_data
            ]
        else:
            self.shard_weights = [1.0 / self.num_shards] * self.num_shards
        logger.info(
            f"All shards trained. Weights: {[f'{w:.3f}' for w in self.shard_weights]}"
        )

    def _aggregate_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregates the embeddings from all sub-models based on the chosen strategy."""
        with torch.no_grad():
            user_embs = [model.get_embeddings()[0] for model in self.shard_models]
            item_embs = [model.get_embeddings()[1] for model in self.shard_models]

        weights = torch.tensor(self.shard_weights, device=self.device).view(-1, 1, 1)

        agg_user_emb = (torch.stack(user_embs) * weights).sum(dim=0)
        agg_item_emb = (torch.stack(item_embs) * weights).sum(dim=0)

        return agg_user_emb, agg_item_emb

    def _create_aggregated_model(self) -> nn.Module:
        """Creates a final model with aggregated embeddings from all sub-models."""
        aggregated_model = get_model(
            self.args, self.data.num_users, self.data.num_items
        ).to(self.device)
        agg_user_emb, agg_item_emb = self._aggregate_embeddings()

        with torch.no_grad():
            aggregated_model.user_emb.weight.data.copy_(agg_user_emb)
            aggregated_model.item_emb.weight.data.copy_(agg_item_emb)

        return aggregated_model

    def _find_affected_shards(self, edges_to_remove: torch.Tensor) -> List[int]:
        """Finds the shards that contain any of the edges marked for unlearning."""
        affected_shards = []
        edges_to_remove_set = set(map(tuple, edges_to_remove.t().tolist()))

        for i, shard_edges in enumerate(self.shard_data):
            shard_edges_set = set(map(tuple, shard_edges.t().tolist()))
            if not edges_to_remove_set.isdisjoint(shard_edges_set):
                affected_shards.append(i)

        logger.info(f"Found {len(affected_shards)} affected shards: {affected_shards}")
        return affected_shards

    def _retrain_affected_shards(
        self, affected_shards: List[int], edges_to_remove: torch.Tensor
    ):
        """Retrains the sub-models for the shards affected by the unlearning request."""
        logger.info(f"Retraining {len(affected_shards)} affected shards...")
        edges_to_remove_set = set(map(tuple, edges_to_remove.t().tolist()))

        for shard_id in affected_shards:
            original_edges = self.shard_data[shard_id].t().tolist()
            remaining_edges = [
                edge
                for edge in original_edges
                if tuple(edge) not in edges_to_remove_set
            ]

            if not remaining_edges:
                logger.warning(
                    f"Shard {shard_id} is empty after unlearning. Reinitializing model."
                )
                self.shard_data[shard_id] = torch.empty((2, 0), dtype=torch.long)
                self.shard_models[shard_id] = get_model(
                    self.args, self.data.num_users, self.data.num_items
                ).to(self.device)
            else:
                new_shard_edges = torch.tensor(remaining_edges, dtype=torch.long).t()
                self.shard_data[shard_id] = new_shard_edges
                self.shard_models[shard_id] = self._train_shard_model(
                    shard_id, new_shard_edges
                )

    def unlearn(self) -> Tuple[nn.Module, torch.Tensor, float]:
        """Executes the SISA unlearning process."""
        logger.info("Executing SISA unlearning process...")
        start_time = time.time()
        edges_to_remove = self.data.edges_to_remove

        if not self.shard_models:
            self._train_all_shards()

        affected_shards = self._find_affected_shards(edges_to_remove)
        if affected_shards:
            self._retrain_affected_shards(affected_shards, edges_to_remove)

        unlearned_model = self._create_aggregated_model()
        unlearning_time = time.time() - start_time
        logger.info(
            f"SISA unlearning complete. Time taken: {unlearning_time:.2f} seconds."
        )

        return unlearned_model, edges_to_remove, unlearning_time
