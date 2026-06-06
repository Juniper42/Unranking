import time

import torch

from logger import Logger
from trainer import train_model
from utils import get_model

logger = Logger.get_logger("Unlearning(Retrain)")


class Retrain:
    """
    Implements the retraining approach to machine unlearning.
    This method discards the original model and trains a new one from scratch on the
    dataset with the forgotten data removed. It serves as the gold standard for
    unlearning effectiveness.
    """

    def __init__(self, model, data, args, device):
        """
        Initializes the Retrain unlearner.

        Args:
            model: The original model (not used, but kept for API consistency).
            data: The data object.
            args: A dictionary of configuration arguments.
            device: The computation device.
        """
        self.model = model
        self.data = data
        self.args = args
        self.device = device

    def unlearn(self, user_train_items=None):
        """
        Executes the retraining process.

        Args:
            user_train_items: Optional dictionary of user's training items for evaluation.

        Returns:
            A tuple of (retrained_model, edges_to_remove, unlearning_time).
        """
        logger.info("Starting unlearning via retraining...")
        start_time = time.time()

        # Create a new model instance with the same architecture
        retrained_model = get_model(
            self.args, self.data.num_users, self.data.num_items
        ).to(self.device)

        # Temporarily replace the original training data with the data after removal
        original_train_edge_index = self.data.train_edge_index
        original_train_ratings = getattr(self.data, "train_ratings", None)
        self.data.train_edge_index = self.data.train_edge_index_after_remove
        if hasattr(self.data, "train_ratings_after_remove"):
            self.data.train_ratings = self.data.train_ratings_after_remove

        logger.info(
            f"Retraining on {self.data.train_edge_index.size(1)} edges after removing {self.data.edges_to_remove.size(1)} edges."
        )

        try:
            # Train the new model from scratch
            train_model(
                model=retrained_model,
                data=self.data,
                args=self.args,
                device=self.device,
                user_train_items=user_train_items,
            )
        finally:
            # Restore the original training data to not affect subsequent evaluations
            self.data.train_edge_index = original_train_edge_index
            if original_train_ratings is not None:
                self.data.train_ratings = original_train_ratings

        unlearning_time = time.time() - start_time
        logger.info(f"Retraining complete. Time taken: {unlearning_time:.2f} seconds.")

        return retrained_model, self.data.edges_to_remove, unlearning_time
