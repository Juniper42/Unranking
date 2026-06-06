import copy
import time

import torch
import torch.nn.functional as F

from logger import Logger
from trainer import train_model
from utils import determine_is_gnn, get_model

logger = Logger.get_logger("Unlearning(UltraRE)")


def _balanced_kmeans(features, n_clusters, n_iter=20):
    """Balanced KMeans: standard KMeans followed by a sort+round assignment
    that enforces (num_samples / n_clusters) members per cluster.
    """
    n, d = features.size()
    target_size = n // n_clusters

    # Init: pick n_clusters samples uniformly at random as centroids.
    idx = torch.randperm(n, device=features.device)[:n_clusters]
    centroids = features[idx].clone()

    labels = torch.zeros(n, dtype=torch.long, device=features.device)
    for _ in range(n_iter):
        d2 = torch.cdist(features, centroids).pow(2)  # n x k

        # Balanced assignment: greedy fill of each cluster up to target_size.
        order = torch.argsort(d2.min(dim=1).values)  # easiest assignments first
        counts = torch.zeros(n_clusters, dtype=torch.long,
                             device=features.device)
        for i in order.tolist():
            ranking = torch.argsort(d2[i])
            for c in ranking.tolist():
                if counts[c] < target_size:
                    labels[i] = c
                    counts[c] += 1
                    break
            else:
                labels[i] = ranking[0]

        # Update centroids.
        new_centroids = torch.stack([
            features[labels == c].mean(0) if (labels == c).any()
            else centroids[c] for c in range(n_clusters)
        ])
        if torch.allclose(new_centroids, centroids, atol=1e-5):
            break
        centroids = new_centroids
    return labels


class UltraRE:
    def __init__(self, model, data, args, device):
        self.model = model
        self.data = data
        self.args = args
        self.device = device
        self.n_group = int(args.get("ultrare_n_group",
                                    args.get("num_partitions", 3)))
        self.sub_epochs = int(args.get("ultrare_sub_epochs",
                                       args.get("sub_epochs", 10)))

    def _user_features(self):
        """Returns one feature vector per user used for clustering.

        The reference repo can cluster either on raw rating rows or on the
        pretrained user embeddings; we follow the cheaper embedding option.
        """
        if determine_is_gnn(self.args.get("backbone")):
            with torch.no_grad():
                u, _ = self.model(edge_index=self.data.train_edge_index)
        else:
            with torch.no_grad():
                u, _ = self.model.get_embeddings()
        return u.detach()

    def _partition_users(self):
        features = self._user_features()
        labels = _balanced_kmeans(features, self.n_group)
        groups = [torch.nonzero(labels == c, as_tuple=False).flatten()
                  for c in range(self.n_group)]
        return groups

    def _train_subgraph(self, group_users):
        """Train a fresh model on the edges incident to `group_users`."""
        retained = self.data.train_edge_index_after_remove
        mask = torch.isin(retained[0], group_users)
        sub_edges = retained[:, mask]

        sub_model = get_model(self.args, self.data.num_users,
                              self.data.num_items).to(self.device)
        sub_data = copy.copy(self.data)
        sub_data.train_edge_index = sub_edges
        sub_data.train_edge_index_after_remove = sub_edges
        sub_args = dict(self.args)
        sub_args["epoch"] = self.sub_epochs
        sub_args["early_stop"] = False
        train_model(sub_model, sub_data, sub_args, self.device)
        return sub_model

    def _aggregate(self, sub_models, groups):
        """Merge per-group user embeddings into one model (Eq. 5 of the
        reference repo). Item embeddings are averaged across groups.
        """
        merged = copy.deepcopy(self.model)
        if hasattr(merged, "clear_cache"):
            merged.clear_cache()
        with torch.no_grad():
            if hasattr(merged, "user_emb"):
                for sm, g in zip(sub_models, groups):
                    merged.user_emb.weight.data[g] = (
                        sm.user_emb.weight.data[g]
                    )
            if hasattr(merged, "item_emb"):
                avg = torch.stack([sm.item_emb.weight.data
                                   for sm in sub_models]).mean(0)
                merged.item_emb.weight.data.copy_(avg)
        return merged

    def unlearn(self):
        logger.info("Executing UltraRE unlearning...")
        start = time.time()

        groups = self._partition_users()
        logger.info(f"UltraRE: {self.n_group} groups, sizes="
                    f"{[g.numel() for g in groups]}")

        # Find groups containing affected users (item unlearning => all).
        affected_users = set(self.data.edges_to_remove[0].cpu().tolist())
        retrain_gids = set()
        for gid, g in enumerate(groups):
            if any(u.item() in affected_users for u in g):
                retrain_gids.add(gid)
        logger.info(f"UltraRE: retraining groups {sorted(retrain_gids)} / "
                    f"{self.n_group}.")

        # Train one sub-model per group; reuse retrained=False groups in
        # principle, but starting from scratch keeps the implementation
        # short (matches the "from-scratch on unlearn" branch in the repo).
        sub_models = [self._train_subgraph(g) for g in groups]

        unlearned = self._aggregate(sub_models, groups)
        elapsed = time.time() - start
        logger.info(f"UltraRE complete. Time: {elapsed:.2f}s.")
        return unlearned, self.data.edges_to_remove, elapsed
