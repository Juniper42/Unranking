import hashlib
import os

import torch

from logger import Logger

logger = Logger.get_logger("ModelCache")


class ModelCache:
    """
    Manages caching of model weights to avoid re-training.
    A unique filename is generated based on the model's hyperparameters.
    """

    def __init__(self, cache_root_dir="./"):
        """
        Initializes the cache manager.

        Args:
            cache_root_dir: The root directory where the .model_cache folder will be created.
        """
        self.cache_dir = os.path.join(cache_root_dir, ".model_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _generate_cache_filename(self, args):
        """
        Generates a unique, stable cache filename based on model hyperparameters.

        Args:
            args: A dictionary of training arguments.

        Returns:
            A string representing the cache filename.
        """
        params = [
            args.get("dataset", "unknown"),
            args.get("backbone", "unknown"),
            args.get("lr", 0.0),
            args.get("epoch", 0),
            args.get("emb_dim", 0),
            args.get("batch_size", 0),
            args.get("num_layers", 0),
        ]
        # Use a hash to keep the filename length manageable
        params_str = "-".join(map(str, params))
        hash_id = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return f"{args.get('dataset')}-{args.get('backbone')}-{hash_id}.pth"

    def _get_cache_path(self, args):
        """
        Constructs the full path to the cache file.

        Args:
            args: A dictionary of training arguments.

        Returns:
            The full path to the potential cache file.
        """
        filename = self._generate_cache_filename(args)
        return os.path.join(self.cache_dir, filename)

    def model_exists(self, args):
        """
        Checks if a cached model with the same parameters already exists.

        Args:
            args: A dictionary of training arguments.

        Returns:
            True if a cached model exists, False otherwise.
        """
        cache_path = self._get_cache_path(args)
        exists = os.path.exists(cache_path)
        if exists:
            logger.info(f"Found cached model: {cache_path}")
        else:
            logger.info(f"No cached model found for the given parameters.")
        return exists

    def save_model(self, model, args):
        """
        Saves the model's state dictionary to the cache.

        Args:
            model: The model to save.
            args: A dictionary of training arguments used to generate the cache filename.
        """
        cache_path = self._get_cache_path(args)
        try:
            torch.save(model.state_dict(), cache_path)
            logger.info(f"Model weights saved to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save model to {cache_path}: {e}")

    def load_model(self, model, args, device):
        """
        Loads model weights from the cache.

        Args:
            model: The model instance to load weights into.
            args: A dictionary of training arguments.
            device: The device to map the loaded weights to.

        Returns:
            The model with loaded weights.
        """
        cache_path = self._get_cache_path(args)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        try:
            state_dict = torch.load(cache_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            logger.info(f"Model weights loaded from cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {cache_path}: {e}")
            raise

        return model

    def get_model_summary(self, args):
        """
        Provides a summary string of the model's key hyperparameters.

        Args:
            args: A dictionary of training arguments.

        Returns:
            A summary string.
        """
        return (
            f"Dataset: {args.get('dataset', 'N/A')}, "
            f"Backbone: {args.get('backbone', 'N/A')}, "
            f"LR: {args.get('lr', 'N/A')}, "
            f"Epochs: {args.get('epoch', 'N/A')}, "
            f"Embedding Dim: {args.get('emb_dim', 'N/A')}, "
            f"Batch Size: {args.get('batch_size', 'N/A')}, "
            f"Layers: {args.get('num_layers', 'N/A')}"
        )
