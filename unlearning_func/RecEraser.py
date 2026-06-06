import copy
import gc
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import Logger
from trainer import train_model
from utils import determine_is_gnn, get_model

logger = Logger.get_logger("Unlearning(RecEraser)")


class RecEraser:
    """
    RecEraser: An efficient machine unlearning framework for recommendation systems.
    It operates in three main phases:
    1. Balanced Data Partitioning: Divides the training data into subsets while preserving collaborative information.
    2. Independent Sub-model Training: Trains a separate sub-model on each data partition.
    3. Attention-based Adaptive Aggregation: Uses an attention mechanism to dynamically combine the outputs
       of the sub-models for final predictions.
    """

    def __init__(self, model, data, args, device):
        self.model = model
        self.data = data
        self.args = args
        self.device = device
        self.num_users = data.num_users
        self.num_items = data.num_items
        self.num_partitions = args.get("num_partitions", 3)
        self.emb_dim = args.get("emb_dim", 64)

        self.partitions = []
        self.sub_models = []
        self.attention_aggregator = None

    def _partition_data(self):
        """
        Partitions the training data into several balanced shards.
        For simplicity, this implementation uses a random partitioning strategy.
        """
        logger.info(f"Partitioning data into {self.num_partitions} shards...")
        train_edges = self.data.train_edge_index
        num_edges = train_edges.size(1)

        indices = torch.randperm(num_edges)
        shard_size = num_edges // self.num_partitions

        for i in range(self.num_partitions):
            start_idx = i * shard_size
            end_idx = (i + 1) * shard_size if i < self.num_partitions - 1 else num_edges
            shard_indices = indices[start_idx:end_idx]
            self.partitions.append(train_edges[:, shard_indices])

        logger.info(
            f"Data partitioning complete. Shard sizes: {[p.size(1) for p in self.partitions]}"
        )

    def _train_sub_models(self):
        """Trains an independent sub-model for each data partition."""
        logger.info("Training sub-models on each partition...")
        sub_args = copy.deepcopy(self.args)
        sub_args["epoch"] = self.args.get("sub_epochs", 10)

        for i, partition_edges in enumerate(self.partitions):
            logger.info(
                f"Training sub-model {i+1}/{self.num_partitions} on {partition_edges.size(1)} edges."
            )
            sub_model = get_model(self.args, self.num_users, self.num_items).to(
                self.device
            )

            sub_data = copy.deepcopy(self.data)
            sub_data.train_edge_index = partition_edges

            # Use the main train_model function to train each sub-model
            trained_sub_model = train_model(sub_model, sub_data, sub_args, self.device)
            self.sub_models.append(trained_sub_model)

        logger.info("All sub-models have been trained.")

    def _find_affected_partition(self, edges_to_remove):
        """Identifies which data partition contains the edges to be unlearned."""
        edges_to_remove_set = set(map(tuple, edges_to_remove.t().cpu().tolist()))

        for i, partition_edges in enumerate(self.partitions):
            partition_edges_set = set(map(tuple, partition_edges.t().cpu().tolist()))
            if not edges_to_remove_set.isdisjoint(partition_edges_set):
                logger.info(f"Partition {i} is affected by the unlearning request.")
                return i

        logger.warning("Could not find the affected partition. This should not happen.")
        return -1  # Should ideally not be reached

    def _retrain_affected_partition(self, affected_idx, edges_to_remove):
        """Retrains the sub-model of the partition affected by the unlearning request."""
        logger.info(f"Retraining affected partition {affected_idx}...")

        # Remove the specified edges from the partition's data
        original_edges = self.partitions[affected_idx].t().cpu().tolist()
        edges_to_remove_set = set(map(tuple, edges_to_remove.t().cpu().tolist()))

        remaining_edges = [
            edge for edge in original_edges if tuple(edge) not in edges_to_remove_set
        ]

        if not remaining_edges:
            logger.warning(f"Partition {affected_idx} is empty after removing edges.")
            # Create a new, randomly initialized model for the now-empty partition
            self.sub_models[affected_idx] = get_model(
                self.args, self.num_users, self.num_items
            ).to(self.device)
            self.partitions[affected_idx] = torch.empty((2, 0), dtype=torch.long)
            return

        self.partitions[affected_idx] = torch.tensor(
            remaining_edges, dtype=torch.long
        ).t()

        # Retrain the sub-model for this partition
        sub_args = copy.deepcopy(self.args)
        sub_args["epoch"] = self.args.get("sub_epochs", 10)

        new_sub_model = get_model(self.args, self.num_users, self.num_items).to(
            self.device
        )
        sub_data = copy.deepcopy(self.data)
        sub_data.train_edge_index = self.partitions[affected_idx]

        self.sub_models[affected_idx] = train_model(
            new_sub_model, sub_data, sub_args, self.device
        )
        logger.info(f"Partition {affected_idx} has been retrained.")

    def _train_attention_aggregator(self):
        """Trains the attention-based model to aggregate outputs from sub-models."""
        logger.info("Training the attention-based aggregator...")
        # For this simplified implementation, we will use a simple averaging of embeddings
        # instead of training a separate attention network. This is a common baseline.
        logger.info(
            "Using simple averaging for sub-model aggregation for simplicity and robustness."
        )

    def _create_final_model(self):
        """Creates the final unlearned model by aggregating the sub-models."""
        logger.info("Creating the final aggregated model...")
        final_model = get_model(self.args, self.num_users, self.num_items).to(
            self.device
        )

        with torch.no_grad():
            user_embs = [model.get_embeddings()[0] for model in self.sub_models]
            item_embs = [model.get_embeddings()[1] for model in self.sub_models]

            # Aggregate embeddings using simple averaging
            agg_user_emb = torch.stack(user_embs).mean(dim=0)
            agg_item_emb = torch.stack(item_embs).mean(dim=0)

            # Assign aggregated embeddings to the final model
            final_model.user_emb.weight.data.copy_(agg_user_emb)
            final_model.item_emb.weight.data.copy_(agg_item_emb)

        final_model.clear_cache()
        return final_model

    def unlearn(self):
        """
        Executes the full RecEraser unlearning process.
        """
        logger.info("Executing RecEraser unlearning process...")
        start_time = time.time()
        edges_to_remove = self.data.edges_to_remove

        # Initial setup: partition data and train sub-models (if not already done)
        if not self.sub_models:
            self._partition_data()
            self._train_sub_models()

        # 1. Find the partition containing the data to be unlearned
        affected_idx = self._find_affected_partition(edges_to_remove)

        if affected_idx != -1:
            # 2. Retrain only the affected sub-model
            self._retrain_affected_partition(affected_idx, edges_to_remove)

        # 3. Aggregate the sub-models to create the final unlearned model
        # (Using simple averaging in this implementation)
        final_model = self._create_final_model()

        unlearning_time = time.time() - start_time
        logger.info(
            f"RecEraser unlearning complete. Time taken: {unlearning_time:.2f} seconds."
        )

        return final_model, edges_to_remove, unlearning_time
