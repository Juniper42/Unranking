# unlearning_func/__init__.py

import time

import torch

from logger import Logger
from trainer import train_model
from utils import get_model

# Import implemented unlearning methods.
from .CertifiedRemoval import CertifiedRemoval
from .GFEraser import GFEraser
from .GSGCF_RU import GSGCF_RU
from .IFRU import IFRU
from .Unranking import Unranking
from .RecEraser import RecEraser
from .RRL import RRL
from .SISA import SISA
from .UltraRE import UltraRE
from .UnlearnRec import UnlearnRec

logger = Logger.get_logger("Unlearning")


def retrain_unlearning(original_model, data, args, device, user_train_items=None):
    """
    Performs unlearning by retraining the model from scratch on the remaining data.
    Serves as the "gold standard" for unlearning effectiveness.
    """
    logger.info("Starting unlearning via retraining...")
    start_time = time.time()
    edges_to_remove = data.edges_to_remove

    new_model = get_model(args, data.num_users, data.num_items).to(device)

    original_train_edge_index = data.train_edge_index
    original_train_ratings = getattr(data, "train_ratings", None)
    data.train_edge_index = data.train_edge_index_after_remove
    if hasattr(data, "train_ratings_after_remove"):
        data.train_ratings = data.train_ratings_after_remove
    try:
        retrained_model = train_model(
            new_model, data, args, device, user_train_items=user_train_items
        )
    finally:
        data.train_edge_index = original_train_edge_index
        if original_train_ratings is not None:
            data.train_ratings = original_train_ratings
    unlearning_time = time.time() - start_time
    logger.info(f"Retraining finished. Time taken: {unlearning_time:.2f} seconds.")
    return retrained_model, edges_to_remove, unlearning_time


def gsgcf_ru_unlearning(original_model, data, args, device):
    """Unlearning via GSGCF-RU (graph-based, LightGCN-only)."""
    logger.info("Starting unlearning via GSGCF-RU...")
    return GSGCF_RU(original_model, data, args, device).unlearn()


def receraser_unlearning(original_model, data, args, device):
    """Unlearning via RecEraser (sharded sub-models with adaptive aggregation)."""
    logger.info("Starting unlearning via RecEraser...")
    return RecEraser(original_model, data, args, device).unlearn()


def unranking_unlearning(original_model, data, args, device):
    """
    Performs preference revision via Unranking.
    Unranking combines LPF p-hop scoping and preference factors with the
    PAPU Hessian-scaled contextual demotion update.
    """
    logger.info("Starting preference revision via Unranking...")
    return Unranking(original_model, data, args, device).unlearn()


def raru_unlearning(original_model, data, args, device):
    """Backward-compatible alias for older scripts."""
    return unranking_unlearning(original_model, data, args, device)


def ifru_unlearning(original_model, data, args, device):
    """Influence-Function-based recommendation unlearning."""
    logger.info("Starting unlearning via IFRU...")
    return IFRU(original_model, data, args, device).unlearn()


def sisa_unlearning(original_model, data, args, device):
    """Sharded, Isolated, Sliced, and Aggregated unlearning."""
    logger.info("Starting unlearning via SISA...")
    return SISA(original_model, data, args, device).unlearn()


def certified_removal_unlearning(original_model, data, args, device):
    """Certified Removal with one-step Newton updates."""
    logger.info("Starting unlearning via Certified Removal...")
    return CertifiedRemoval(original_model, data, args, device).unlearn()


def ultrare_unlearning(original_model, data, args, device):
    """
    UltraRE: ultra-fast recommendation unlearning via embedding alignment.
    Re-aligns affected user/item embeddings to the centroid of their
    remaining neighbors instead of retraining.
    """
    logger.info("Starting unlearning via UltraRE...")
    return UltraRE(original_model, data, args, device).unlearn()


def rrl_unlearning(original_model, data, args, device):
    """
    RRL: Rank-Restricted Learning. A short fine-tuning pass on the retained
    interactions, restricted to embeddings of users/items touched by removal.
    """
    logger.info("Starting unlearning via RRL...")
    return RRL(original_model, data, args, device).unlearn()


def unlearnrec_unlearning(original_model, data, args, device):
    """
    UnlearnRec: localized fine-tuning on the k-hop subgraph around target
    interactions with a forgetting loss on D_t.
    """
    logger.info("Starting unlearning via UnlearnRec...")
    return UnlearnRec(original_model, data, args, device).unlearn()


def gferaser_unlearning(original_model, data, args, device):
    """
    GFEraser: graph-flow based eraser. Reweights the message passing on
    the removed edges to zero and fine-tunes briefly on the retained graph.
    """
    logger.info("Starting unlearning via GFEraser...")
    return GFEraser(original_model, data, args, device).unlearn()
