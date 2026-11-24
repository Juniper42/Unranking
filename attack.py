import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from logger import Logger
from models import MIAttackModel
from utils import determine_is_gnn

logger = Logger.get_logger("MIAttack")


class MembershipInferenceAttack:
    """Membership-inference probe used to compute the Revision Rate (RR).

    The paper uses the MIA probe as a *probe for the residual preference
    imprint*, not as a privacy claim.  The protocol follows Section 5.1.6:

    * members are the target interactions ``D_t``;
    * an equal number of non-members is sampled from the validation and test
      interactions of the same users, matched by item-popularity group
      computed on the retained training data;
    * the interaction pairs are split 6:2:2 into attack-training, validation,
      and test sets, with class balance preserved in each split;
    * the probe is a three-layer MLP (8 -> 64 -> 32 -> 1) trained on the
      pre-revision model outputs with BCE and Adam, batch size 256, for up to
      100 epochs with early stopping after 10 epochs without validation-AUROC
      improvement;
    * the trained probe and feature preprocessing are frozen and applied to
      the same test pairs for the revised model;
    * ``RR = symmetric-AUC(M_p) / symmetric-AUC(M')``, where the symmetric AUC
      is ``max(AUC, 1 - AUC)``.
    """

    def __init__(self, pre_revision_model, revised_model, data, device, args):
        self.pre_revision_model = pre_revision_model
        self.revised_model = revised_model
        self.data = data
        self.device = device
        self.args = args

        self.seed = int(args.get("target_seed", args.get("random_seed", 42)))
        self.attack_batch_size = int(args.get("attack_batch_size", 256))
        self.attack_epochs = int(args.get("attack_epochs", 100))
        self.attack_lr = float(args.get("attack_lr", 1e-3))
        self.attack_patience = int(args.get("attack_patience", 10))
        self.max_pairs = int(args.get("attack_max_pairs", 0))
        self.popularity_bins = int(args.get("mia_popularity_bins", 10))

        self.probe = None
        self.feature_mean = None
        self.feature_std = None
        self.test_features = None
        self.test_labels = None
        self.num_members = 0
        self.num_non_members = 0
        self.attack_train_size = 0

    # ------------------------------------------------------------------
    # Pair construction
    # ------------------------------------------------------------------
    def _canonical_pairs(self, edges):
        """Return unique ``(user, local_item)`` pairs from forward/reverse edges."""
        if edges is None or edges.numel() == 0:
            return []
        pairs = set()
        item_lo = self.data.num_users
        item_hi = self.data.num_users + self.data.num_items
        for src, dst in edges.t().detach().cpu().tolist():
            if 0 <= src < self.data.num_users and item_lo <= dst < item_hi:
                pairs.add((int(src), int(dst - self.data.num_users)))
            elif item_lo <= src < item_hi and 0 <= dst < self.data.num_users:
                pairs.add((int(dst), int(src - self.data.num_users)))
        return sorted(pairs)

    def _popularity_groups(self):
        """Assign each item to an item-popularity decile on retained training data."""
        ratings = getattr(self.data, "train_ratings_after_remove", None)
        if ratings is None or ratings.empty:
            ratings = getattr(self.data, "train_ratings", None)
        counts = np.zeros(self.data.num_items, dtype=np.int64)
        if ratings is not None and not ratings.empty:
            np.add.at(
                counts,
                ratings["item_idx"].astype(int).to_numpy(),
                1,
            )

        # Rank items by popularity and cut the ranking into equal-frequency
        # groups.  Unseen items form their own group (-1).
        order = np.argsort(counts, kind="stable")
        ranks = np.empty(self.data.num_items, dtype=np.int64)
        ranks[order] = np.arange(self.data.num_items)
        groups = np.floor(
            ranks / max(self.data.num_items - 1, 1) * self.popularity_bins
        ).astype(np.int64)
        groups = np.clip(groups, 0, max(self.popularity_bins - 1, 0))
        groups[counts == 0] = -1
        return groups, counts

    def _validation_test_items_by_user(self):
        by_user = {}
        for attr in ("val_ratings", "test_ratings"):
            ratings = getattr(self.data, attr, None)
            if ratings is None or ratings.empty:
                continue
            for u, i in zip(
                ratings["user_idx"].astype(int).tolist(),
                ratings["item_idx"].astype(int).tolist(),
            ):
                by_user.setdefault(int(u), []).append(int(i))
        return by_user

    def _build_member_nonmember_pairs(self, target_edges):
        """Build ``D_t`` members and popularity-matched non-members."""
        members = self._canonical_pairs(target_edges)
        if not members:
            logger.warning("D_t is empty; the MIA probe cannot be built.")
            return [], []

        if 0 < self.max_pairs < len(members):
            rng = random.Random(self.seed)
            members = rng.sample(members, self.max_pairs)
            logger.info(
                "Capped MIA members to %d interactions for tractability "
                "(explicit runtime safeguard; the paper uses all pairs).",
                len(members),
            )

        groups, _ = self._popularity_groups()
        val_test_by_user = self._validation_test_items_by_user()
        # Validation and test interactions are disjoint from every training
        # set by construction, so they are valid non-members without
        # materialising the 100M-interaction training pair set.
        rng = random.Random(self.seed + 1)

        non_members = set()
        unmatched = 0
        for user, item in members:
            candidates = [
                candidate
                for candidate in val_test_by_user.get(user, [])
                if (user, candidate) not in non_members
            ]
            if not candidates:
                unmatched += 1
                continue

            target_group = int(groups[item])
            matched = [
                candidate
                for candidate in candidates
                if int(groups[candidate]) == target_group
            ]
            pool = matched if matched else candidates
            non_members.add((user, rng.choice(pool)))
            if len(non_members) >= len(members):
                break

        non_members = sorted(non_members)
        if unmatched:
            logger.warning(
                "No validation/test non-member was available for %d target "
                "interactions; they are excluded from the probe.",
                unmatched,
            )
        if not non_members:
            logger.warning(
                "Could not sample non-members from validation/test data. "
                "Set val_size > 0 so the MIA protocol can be applied."
            )
            return [], []

        # Keep the two classes balanced.
        size = min(len(members), len(non_members))
        if len(members) > size:
            members = rng.sample(members, size)
        if len(non_members) > size:
            non_members = rng.sample(non_members, size)
        return members, non_members

    def _split_pairs(self, members, non_members):
        """Split each class 6:2:2 and concatenate the splits."""
        rng = random.Random(self.seed + 2)

        def split_class(pairs):
            pairs = list(pairs)
            rng.shuffle(pairs)
            n = len(pairs)
            if n == 0:
                return [], [], []
            if n < 5:
                # Degenerate split for tiny probes; kept disjoint.
                n_train = max(1, n - 2)
                n_val = 1 if n - n_train >= 2 else 0
                return pairs[:n_train], pairs[n_train : n_train + n_val], pairs[n_train + n_val :]
            n_train = int(round(0.6 * n))
            n_val = int(round(0.2 * n))
            n_train = min(max(n_train, 1), n)
            n_val = min(n_val, n - n_train)
            return pairs[:n_train], pairs[n_train : n_train + n_val], pairs[n_train + n_val :]

        m_train, m_val, m_test = split_class(members)
        n_train, n_val, n_test = split_class(non_members)
        train_pairs = m_train + n_train
        val_pairs = m_val + n_val
        test_pairs = m_test + n_test
        train_labels = [1] * len(m_train) + [0] * len(n_train)
        val_labels = [1] * len(m_val) + [0] * len(n_val)
        test_labels = [1] * len(m_test) + [0] * len(n_test)
        return (
            (train_pairs, train_labels),
            (val_pairs, val_labels),
            (test_pairs, test_labels),
        )

    # ------------------------------------------------------------------
    # Single-model posterior features
    # ------------------------------------------------------------------
    def _graph_for_model(self, model):
        if model is self.pre_revision_model:
            return getattr(
                self.data, "pre_revision_edge_index", self.data.train_edge_index
            )
        return getattr(
            self.data,
            "eval_edge_index",
            getattr(self.data, "train_edge_index_after_remove", self.data.train_edge_index),
        )

    @torch.no_grad()
    def _embeddings_for_model(self, model, graph_edge_index):
        model.eval()
        if hasattr(model, "clear_cache"):
            model.clear_cache()
        if determine_is_gnn(self.args.get("backbone")):
            user_emb, item_emb = model(edge_index=graph_edge_index.to(self.device))
        else:
            user_emb, item_emb = model.get_embeddings()
        return user_emb, item_emb

    @torch.no_grad()
    def _features_for_pairs(self, model, pairs):
        """Eight distance-based features extracted from model posteriors.

        The features combine the interaction posterior with embedding-level
        cosine, Euclidean, and L1 distances, plus distributional distances
        (entropy and deviation from chance).  They are computed from a single
        model so that the same frozen probe can be applied to every variant.
        """
        if not pairs:
            return torch.empty((0, 8), dtype=torch.float, device=self.device)

        users = torch.tensor(
            [u for u, _ in pairs], dtype=torch.long, device=self.device
        )
        items = torch.tensor(
            [i for _, i in pairs], dtype=torch.long, device=self.device
        )
        graph_edge_index = self._graph_for_model(model)
        user_emb, item_emb = self._embeddings_for_model(model, graph_edge_index)
        if determine_is_gnn(self.args.get("backbone")):
            scores = torch.sum(user_emb[users] * item_emb[items], dim=1).cpu()
        else:
            scores = model(user_indices=users, item_indices=items).detach().cpu()
        u_emb = user_emb[users].detach().cpu()
        i_emb = item_emb[items].detach().cpu()

        eps = 1e-8
        posterior = torch.sigmoid(scores).clamp(eps, 1.0 - eps)
        cosine = F.cosine_similarity(u_emb, i_emb, dim=1)
        euclidean = torch.sqrt(((u_emb - i_emb) ** 2).sum(dim=1) + eps)
        l1 = (u_emb - i_emb).abs().sum(dim=1)
        entropy = -(
            posterior * torch.log(posterior)
            + (1.0 - posterior) * torch.log(1.0 - posterior)
        )
        chance_distance = (posterior - 0.5).abs()

        features = torch.stack(
            [
                posterior,
                scores,
                1.0 - cosine,
                euclidean,
                l1,
                entropy,
                chance_distance,
                torch.log1p(euclidean),
            ],
            dim=1,
        )
        return features.detach().cpu()

    # ------------------------------------------------------------------
    # Probe training and evaluation
    # ------------------------------------------------------------------
    def _standardize(self, train_features, *other_features):
        self.feature_mean = train_features.mean(dim=0, keepdim=True)
        self.feature_std = train_features.std(dim=0, keepdim=True) + 1e-6
        standardized = [(train_features - self.feature_mean) / self.feature_std]
        for features in other_features:
            standardized.append((features - self.feature_mean) / self.feature_std)
        return standardized

    def _fit_probe(self, train_features, train_labels, val_features, val_labels):
        # Freeze the probe initialization and mini-batch order so that the
        # same probe is used for every revised model variant.
        torch.manual_seed(self.seed + 100)
        np.random.seed(self.seed + 100)
        random.seed(self.seed + 100)
        self.probe = MIAttackModel(train_features.size(1)).to(self.device)
        optimizer = torch.optim.Adam(self.probe.parameters(), lr=self.attack_lr)
        criterion = nn.BCEWithLogitsLoss()

        train_features = train_features.to(self.device)
        train_labels = train_labels.to(self.device).float()
        val_features = val_features.to(self.device)
        val_labels_np = val_labels.cpu().numpy()

        best_auc = -1.0
        best_state = None
        patience = 0
        for epoch in range(self.attack_epochs):
            self.probe.train()
            perm = torch.randperm(train_features.size(0), device=self.device)
            total_loss = 0.0
            num_batches = 0
            for start in range(0, train_features.size(0), self.attack_batch_size):
                indices = perm[start : start + self.attack_batch_size]
                optimizer.zero_grad()
                logits = self.probe(train_features[indices]).view(-1)
                loss = criterion(logits, train_labels[indices])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.probe.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            self.probe.eval()
            with torch.no_grad():
                val_scores = (
                    torch.sigmoid(self.probe(val_features).view(-1)).cpu().numpy()
                )
            if len(np.unique(val_labels_np)) > 1:
                val_auc = roc_auc_score(val_labels_np, val_scores)
            else:
                val_auc = 0.5

            if val_auc > best_auc + 1e-6:
                best_auc = val_auc
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.probe.state_dict().items()
                }
                patience = 0
            else:
                patience += 1

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "MIA probe epoch %d/%d: loss=%.4f, val AUROC=%.4f",
                    epoch + 1,
                    self.attack_epochs,
                    total_loss / max(num_batches, 1),
                    val_auc,
                )
            if patience >= self.attack_patience:
                logger.info(
                    "MIA probe early stopping at epoch %d (best val AUROC=%.4f).",
                    epoch + 1,
                    best_auc,
                )
                break

        if best_state is not None:
            self.probe.load_state_dict(best_state)
        self.probe.eval()
        return best_auc

    @torch.no_grad()
    def _evaluate_probe(self, features, labels):
        features = features.to(self.device)
        labels_np = labels.cpu().numpy()
        probabilities = (
            torch.sigmoid(self.probe(features).view(-1)).cpu().numpy()
        )
        predictions = (probabilities > 0.5).astype(int)
        auc = (
            roc_auc_score(labels_np, probabilities)
            if len(np.unique(labels_np)) > 1
            else 0.5
        )
        metrics = {
            "accuracy": accuracy_score(labels_np, predictions),
            "precision": precision_score(labels_np, predictions, zero_division=0),
            "recall": recall_score(labels_np, predictions, zero_division=0),
            "f1_score": f1_score(labels_np, predictions, zero_division=0),
            "auc": auc,
            "symmetric_auc": max(auc, 1.0 - auc),
            "average_precision": (
                average_precision_score(labels_np, probabilities)
                if len(np.unique(labels_np)) > 1
                else 0.0
            ),
        }
        return metrics

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run_attack(self, target_edges=None):
        """Train the probe on ``M_p`` and evaluate it on ``M'``.

        Returns a dictionary containing the pre- and post-revision AUCs and
        the Revision Rate ``RR = symmetric-AUC(M_p) / symmetric-AUC(M')``.
        """
        logger.info(
            "Starting the membership-inference probe for RR "
            "(preference-imprint probe, not a privacy claim)."
        )
        start = time.time()
        if target_edges is None:
            target_edges = getattr(self.data, "edges_to_remove", None)
        if target_edges is None or target_edges.numel() == 0:
            logger.error("No target set D_t provided for the MIA probe.")
            return None

        members, non_members = self._build_member_nonmember_pairs(target_edges)
        if not members or not non_members:
            return None
        self.num_members = len(members)
        self.num_non_members = len(non_members)

        (
            (train_pairs, train_labels),
            (val_pairs, val_labels),
            (test_pairs, test_labels),
        ) = self._split_pairs(members, non_members)
        if not train_pairs or not val_pairs or not test_pairs:
            logger.warning(
                "MIA split is empty (train=%d, val=%d, test=%d); "
                "the probe cannot be trained.",
                len(train_pairs),
                len(val_pairs),
                len(test_pairs),
            )
            return None
        self.attack_train_size = len(train_pairs)
        self.test_pairs = test_pairs
        self.test_labels = torch.tensor(test_labels, dtype=torch.float)

        logger.info(
            "MIA pairs: %d members, %d non-members; split %d/%d/%d.",
            self.num_members,
            self.num_non_members,
            len(train_pairs),
            len(val_pairs),
            len(test_pairs),
        )

        # Pre-revision features and frozen preprocessing.
        train_features = self._features_for_pairs(self.pre_revision_model, train_pairs)
        val_features = self._features_for_pairs(self.pre_revision_model, val_pairs)
        test_features = self._features_for_pairs(self.pre_revision_model, test_pairs)
        train_features, val_features, test_features = self._standardize(
            train_features, val_features, test_features
        )
        self.test_features = test_features
        self._fit_probe(
            train_features,
            torch.tensor(train_labels, dtype=torch.float),
            val_features,
            torch.tensor(val_labels, dtype=torch.float),
        )
        pre_metrics = self._evaluate_probe(test_features, self.test_labels)

        results = {
            "auroc_pre_revision": pre_metrics["auc"],
            "symmetric_auroc_pre_revision": pre_metrics["symmetric_auc"],
            "num_members": self.num_members,
            "num_non_members": self.num_non_members,
            "attack_train_size": self.attack_train_size,
            "attack_test_size": len(test_pairs),
        }

        if self.revised_model is not None:
            revised_features = self._features_for_pairs(
                self.revised_model, test_pairs
            )
            revised_features = (
                revised_features - self.feature_mean
            ) / self.feature_std
            post_metrics = self._evaluate_probe(revised_features, self.test_labels)
            rr = (
                pre_metrics["symmetric_auc"] / post_metrics["symmetric_auc"]
                if post_metrics["symmetric_auc"] > 0
                else float("inf")
            )
            results.update(
                {
                    "auroc_post_revision": post_metrics["auc"],
                    "symmetric_auroc_post_revision": post_metrics["symmetric_auc"],
                    "RR": rr,
                    "accuracy": post_metrics["accuracy"],
                    "precision": post_metrics["precision"],
                    "recall": post_metrics["recall"],
                    "f1_score": post_metrics["f1_score"],
                    "auc": post_metrics["auc"],
                    "average_precision": post_metrics["average_precision"],
                }
            )
            logger.info(
                "MIA-AUC(M_p)=%.4f, MIA-AUC(M')=%.4f, "
                "symmetric AUC %.4f -> %.4f, RR=%.4f.",
                pre_metrics["auc"],
                post_metrics["auc"],
                pre_metrics["symmetric_auc"],
                post_metrics["symmetric_auc"],
                rr,
            )
        else:
            results.update(
                {
                    "auroc_post_revision": None,
                    "symmetric_auroc_post_revision": None,
                    "RR": None,
                    "auc": pre_metrics["auc"],
                }
            )
            logger.info(
                "MIA-AUC(M_p)=%.4f (no revised model supplied).",
                pre_metrics["auc"],
            )

        logger.info(
            "MIA probe finished in %.2fs.", time.time() - start
        )
        return results
