import copy
import math
import time

import torch
import torch.nn.functional as F
from torch.autograd import grad

from logger import Logger
from utils import determine_is_gnn

logger = Logger.get_logger("Unlearning(Unranking)")


class Unranking:
    """Unranking for preference revision.

    This class implements the two components of Unranking described in the
    manuscript *Fast Unranking for Preference Revision in Recommendation*:

    * the Localized Preference Factor (LPF), which builds the ``p``-hop
      influence scope ``D_inf`` around the target set ``D_t`` and assigns a
      preference factor ``omega(v)`` to every entity in that scope
      (Algorithm 1, Stages 1--2), and
    * the Preference-Aware Parameter Update (PAPU), which builds the
      aggregated demotion gradient ``g`` and solves
      ``(H_inf + lambda I) DeltaTheta = g`` with Conjugate Gradient and
      Hessian-vector products (Algorithm 1, Stage 3).

    Notation follows the paper: ``D`` is the interaction set, ``D_t`` the
    target (outdated) set, ``D_r = D \\ D_t`` the retained set, ``D_inf`` the
    influence scope, and ``Theta`` / ``Theta'`` the pre-revision and revised
    parameters.
    """

    def __init__(self, model, data, args, device):
        self.model = model
        self.data = data
        self.args = args
        self.device = device

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.num_users = data.num_users
        self.num_items = data.num_items
        self.is_gnn = determine_is_gnn(args.get("backbone"))
        self.backbone = str(args.get("backbone", "")).lower()

        # Paper hyperparameters (Section 5.1.6): propagation depth p,
        # Conjugate Gradient iteration count T, damping factor lambda.
        self.p = int(args.get("unranking_hops", 2))
        self.T = int(args.get("unranking_cg_steps", 30))
        self.lam = float(args.get("unranking_damping_lambda", 1e-2))
        self.max_scope_interactions = int(
            args.get("unranking_max_scope_interactions", 0)
        )
        if self.p < 0:
            raise ValueError("unranking_hops (p) must be non-negative.")
        if self.T < 1:
            raise ValueError("unranking_cg_steps (T) must be positive.")
        if self.lam < 0:
            raise ValueError("unranking_damping_lambda (lambda) must be >= 0.")

        # Ablation switches corresponding to Table 6 of the paper.
        self.no_scoping = bool(args.get("unranking_no_scoping", False))
        self.no_preference_factor = bool(
            args.get("unranking_no_preference_factor", False)
        )
        self.no_structural = bool(args.get("unranking_no_structural", False))
        self.no_semantic = bool(args.get("unranking_no_semantic", False))
        self.no_contextual_correction = bool(
            args.get("unranking_no_contextual_correction", False)
        )

        # Optional privacy-oriented Gaussian-noise variant (Eq. 16).  The
        # paper reports it only as an optional variant; it is off by default.
        self.use_dp = bool(args.get("unranking_dp", False))
        self.dp_epsilon = float(args.get("unranking_dp_epsilon", 1.0))
        self.dp_delta = float(args.get("unranking_dp_delta", 1e-5))
        self.dp_clip_norm = float(args.get("unranking_dp_clip_norm", 1.0))
        if self.use_dp:
            if self.dp_epsilon <= 0 or not (0 < self.dp_delta < 1):
                raise ValueError(
                    "The DP variant requires epsilon > 0 and 0 < delta < 1."
                )
            if self.dp_clip_norm <= 0:
                raise ValueError("unranking_dp_clip_norm (C') must be positive.")

    # ------------------------------------------------------------------
    # Edge and embedding helpers
    # ------------------------------------------------------------------
    def _empty_edges(self):
        return torch.empty((2, 0), dtype=torch.long, device=self.device)

    def _canonical_edge_tensor(self, edge_index, deduplicate=True):
        """Return unique forward ``(user, global_item_node)`` edges.

        GNN graphs are stored with both forward and reverse edges.  ``D`` in
        the paper is the set of user--item interactions, so reverse edges are
        folded back and duplicate pairs are removed when requested.
        """
        if edge_index is None or edge_index.numel() == 0:
            return self._empty_edges()

        edge_index = edge_index.to(self.device)
        src, dst = edge_index[0], edge_index[1]
        item_lo = self.num_users
        item_hi = self.num_users + self.num_items

        forward = (
            (src >= 0)
            & (src < self.num_users)
            & (dst >= item_lo)
            & (dst < item_hi)
        )
        reverse = (
            (src >= item_lo)
            & (src < item_hi)
            & (dst >= 0)
            & (dst < self.num_users)
        )

        users = torch.cat([src[forward], dst[reverse]])
        items = torch.cat([dst[forward], src[reverse]])

        if users.numel() == 0:
            # Defensive fallback for legacy tensors with local item ids.
            legacy = (
                (src >= 0)
                & (src < self.num_users)
                & (dst >= 0)
                & (dst < self.num_items)
            )
            users = src[legacy]
            items = dst[legacy] + self.num_users

        if users.numel() == 0:
            return self._empty_edges()

        if not deduplicate:
            return torch.stack([users, items], dim=0)

        seen = set()
        pairs = []
        for u, i in zip(users.detach().cpu().tolist(), items.detach().cpu().tolist()):
            key = (int(u), int(i))
            if key not in seen:
                seen.add(key)
                pairs.append(key)
        return self._edge_tensor_from_pairs(pairs)

    def _edge_tensor_from_pairs(self, pairs):
        if not pairs:
            return self._empty_edges()
        return torch.tensor(pairs, dtype=torch.long, device=self.device).t().contiguous()

    def _interaction_dataset_edges(self):
        """Return ``D`` as unique forward user--item edges.

        ``train_edge_index_original`` is the forward-only training graph for
        GNN backbones, so it can be used directly.  This avoids materialising
        a Python set over all interactions on the 100M-interaction datasets.
        """
        source_edges = getattr(
            self.data, "train_edge_index_original", self.data.train_edge_index
        )
        if source_edges is None or source_edges.numel() == 0:
            return self._empty_edges()

        source_edges = source_edges.to(self.device)
        if int(source_edges[1].min().item()) >= self.num_users:
            return source_edges
        return self._canonical_edge_tensor(source_edges, deduplicate=True)

    def _target_edges(self):
        """Return ``D_t`` as canonical forward user--item edges."""
        return self._canonical_edge_tensor(self.data.edges_to_remove, deduplicate=True)

    def _forward_embeddings(self, graph_edge_index=None):
        if self.is_gnn:
            if graph_edge_index is None:
                graph_edge_index = self.data.train_edge_index
            if hasattr(self.model, "clear_cache"):
                self.model.clear_cache()
            return self.model(edge_index=graph_edge_index.to(self.device))
        return self.model.get_embeddings()

    def _entity_embedding(self, node_id, user_emb, item_emb):
        if node_id < self.num_users:
            return user_emb[node_id]
        return item_emb[node_id - self.num_users]

    # ------------------------------------------------------------------
    # Stage 1: p-hop influence scoping (paper Section 4.1.1)
    # ------------------------------------------------------------------
    def _p_hop_influence_scope(self):
        """Construct ``D_inf`` by ``p``-hop propagation (Algorithm 1).

        ``S^(0) = D_t``.  At each hop every interaction sharing a user or an
        item with the current scope is added, i.e.
        ``N(u, i) = {(u', i') in D | u' = u or i' = i}``.  The implementation
        uses boolean entity masks instead of Python neighbourhood lists so
        that it scales to the 100M-interaction datasets.
        """
        all_edges = self._interaction_dataset_edges()
        target_edges = self._target_edges()
        if target_edges.numel() == 0:
            return self._empty_edges(), target_edges, self._empty_edges()

        if self.no_scoping:
            # Ablation "w/o Scoping": the update spans the whole dataset.
            scope_edges = all_edges
        elif self.p == 0:
            # With p = 0 the scope keeps only the target interactions.
            scope_edges = target_edges
        else:
            src = all_edges[0]
            dst = all_edges[1]
            dst_item = dst - self.num_users

            user_mask = torch.zeros(self.num_users, dtype=torch.bool, device=self.device)
            item_mask = torch.zeros(self.num_items, dtype=torch.bool, device=self.device)
            user_mask[target_edges[0]] = True
            item_mask[target_edges[1] - self.num_users] = True

            scope_mask = torch.zeros(all_edges.size(1), dtype=torch.bool, device=self.device)
            for _ in range(self.p):
                incident = user_mask[src] | item_mask[dst_item]
                newly_added = incident & ~scope_mask
                if not bool(newly_added.any()):
                    break
                scope_mask |= incident
                user_mask[src[newly_added]] = True
                item_mask[dst_item[newly_added]] = True

            scope_edges = all_edges[:, scope_mask]

        # Remove ``D_t`` from ``D_inf`` to obtain the related interactions
        # ``D_inf \\ D_t`` used by the contextual correction term.
        if scope_edges.numel() == 0:
            related_edges = self._empty_edges()
        else:
            scope_keys = scope_edges[0] * self.num_items + (
                scope_edges[1] - self.num_users
            )
            target_keys = target_edges[0] * self.num_items + (
                target_edges[1] - self.num_users
            )
            related_edges = scope_edges[:, ~torch.isin(scope_keys, target_keys)]

        # Optional runtime safeguard.  The paper does not cap the scope
        # (``--unranking_max_scope_interactions 0``), so this is disabled by
        # default and remains explicit when enabled.
        if (
            self.max_scope_interactions > 0
            and scope_edges.size(1) > self.max_scope_interactions
        ):
            target_keys = target_edges[0] * self.num_items + (
                target_edges[1] - self.num_users
            )
            scope_keys = scope_edges[0] * self.num_items + (
                scope_edges[1] - self.num_users
            )
            target_in_scope = torch.isin(scope_keys, target_keys)
            target_indices = torch.nonzero(target_in_scope, as_tuple=False).view(-1)
            other_indices = torch.nonzero(~target_in_scope, as_tuple=False).view(-1)
            remaining = max(self.max_scope_interactions - target_indices.numel(), 0)
            keep = torch.cat([target_indices, other_indices[:remaining]])
            scope_edges = scope_edges[:, keep]
            if related_edges.numel() > 0:
                related_edges = related_edges[
                    :, torch.isin(
                        related_edges[0] * self.num_items
                        + (related_edges[1] - self.num_users),
                        scope_edges[0] * self.num_items
                        + (scope_edges[1] - self.num_users),
                    )
                ]
            logger.warning(
                "Influence scope capped to %d interactions; this runtime "
                "safeguard is not part of the paper setting.",
                scope_edges.size(1),
            )

        return scope_edges, target_edges, related_edges

    # ------------------------------------------------------------------
    # Stage 2: preference factor estimation (paper Section 4.1.2)
    # ------------------------------------------------------------------
    def _structural_influence(self, scope_edges):
        """Compute ``I_struct(v) = log(1 + deg_v)`` on ``D_inf`` (Eq. 6).

        The degree is induced by the interactions in ``D_inf`` and is computed
        with ``bincount`` so that the operation remains linear in ``|D_inf|``.
        """
        num_nodes = self.num_users + self.num_items
        nodes = torch.cat([scope_edges[0], scope_edges[1]])
        degree = torch.bincount(nodes, minlength=num_nodes)
        entities = torch.nonzero(degree, as_tuple=False).view(-1)
        values = torch.log1p(degree[entities].float())
        return {
            int(v): float(value)
            for v, value in zip(entities.cpu().tolist(), values.cpu().tolist())
        }

    def _semantic_influence(self, scope_edges, target_entities):
        """Compute the mean cosine similarity to target entities (Eq. 7).

        ``I_sem(v) = e_v^T (sum_e e_e / ||e_e||) / (|E_t| ||e_v||)``.  The
        sum of normalized target embeddings makes the computation linear in
        the number of entities instead of forming a ``|V_inf| x |E_t|``
        similarity matrix.
        """
        if scope_edges.numel() == 0 or not target_entities:
            return {}

        num_nodes = self.num_users + self.num_items
        entity_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        entity_mask[scope_edges[0]] = True
        entity_mask[scope_edges[1]] = True
        entities = torch.nonzero(entity_mask, as_tuple=False).view(-1)

        with torch.no_grad():
            user_emb, item_emb = self._forward_embeddings(self.data.train_edge_index)
            all_emb = torch.cat([user_emb, item_emb], dim=0)
            all_emb = F.normalize(all_emb, dim=1)

            target_index = torch.tensor(
                sorted(target_entities), dtype=torch.long, device=self.device
            )
            target_sum = all_emb[target_index].sum(dim=0)
            values = all_emb[entities] @ target_sum / target_index.numel()

        return {
            int(v): float(value)
            for v, value in zip(entities.cpu().tolist(), values.cpu().tolist())
        }

    def _kernel_calibrate(self, values, target_entities):
        """Gaussian-kernel calibration ``K(x)`` of Eq. 8.

        ``mu_t`` is the mean of the signal over the target entities ``E_t`` and
        ``h`` is its standard deviation over the influence scope ``V_inf``.
        The result maps each signal to ``[0, 1]``.
        """
        if not values:
            return {}

        keys = sorted(values)
        value_tensor = torch.tensor(
            [values[k] for k in keys], dtype=torch.float, device=self.device
        )
        target_values = [values[e] for e in target_entities if e in values]
        if target_values:
            mu_t = torch.tensor(
                target_values, dtype=torch.float, device=self.device
            ).mean()
        else:
            mu_t = value_tensor.mean()
        bandwidth = value_tensor.std(unbiased=False)
        if float(bandwidth.detach()) < 1e-12:
            calibrated = torch.ones_like(value_tensor)
        else:
            calibrated = torch.exp(
                -((value_tensor - mu_t) ** 2) / (2.0 * bandwidth ** 2)
            )
        return {k: float(v) for k, v in zip(keys, calibrated.detach().cpu().tolist())}

    def _compute_preference_factors(self, scope_edges, target_edges):
        """Compute ``omega(v) = K_struct(I_struct(v)) + K_sem(I_sem(v))``.

        The two signals are measured relative to the target entities and are
        calibrated onto a common ``[0, 1]`` scale before summation (Eq. 9).
        """
        if scope_edges.numel() == 0:
            return {}

        entities = sorted(
            set(scope_edges[0].cpu().tolist()) | set(scope_edges[1].cpu().tolist())
        )
        target_entities = sorted(
            set(target_edges[0].cpu().tolist()) | set(target_edges[1].cpu().tolist())
        )

        if self.no_preference_factor or (self.no_structural and self.no_semantic):
            # Ablation "w/o Preference Factor" (uniform weights).
            return {v: 1.0 for v in entities}

        i_struct = (
            self._structural_influence(scope_edges) if not self.no_structural else {}
        )
        i_sem = (
            self._semantic_influence(scope_edges, target_entities)
            if not self.no_semantic
            else {}
        )

        omega_struct = (
            self._kernel_calibrate(i_struct, target_entities) if i_struct else {}
        )
        omega_sem = self._kernel_calibrate(i_sem, target_entities) if i_sem else {}

        preference = {}
        for v in entities:
            value = 0.0
            if not self.no_structural:
                value += omega_struct.get(v, 0.0)
            if not self.no_semantic:
                value += omega_sem.get(v, 0.0)
            preference[v] = float(value)
        return preference

    def _interaction_weights(self, edges, preference):
        """Weight an interaction by the mean factor of its user and item.

        ``w(u, i) = (omega(u) + omega(i)) / 2`` as in Eq. 12.
        """
        if edges.numel() == 0:
            return torch.empty(0, dtype=torch.float, device=self.device)
        if self.no_preference_factor:
            return torch.ones(edges.size(1), dtype=torch.float, device=self.device)

        num_nodes = self.num_users + self.num_items
        pref_tensor = torch.zeros(num_nodes, dtype=torch.float, device=self.device)
        if preference:
            entity_ids = torch.tensor(
                list(preference.keys()), dtype=torch.long, device=self.device
            )
            entity_values = torch.tensor(
                list(preference.values()), dtype=torch.float, device=self.device
            )
            pref_tensor[entity_ids] = entity_values
        return 0.5 * (pref_tensor[edges[0]] + pref_tensor[edges[1]])

    # ------------------------------------------------------------------
    # Stage 3: preference-aware parameter update (paper Section 4.2)
    # ------------------------------------------------------------------
    def _zero_loss(self):
        return sum((p.sum() * 0.0) for p in self.params)

    def _uses_bpr_loss(self):
        # Paper Section 5.1.6: BPR for LightGCN, BCE for WMF and NeuMF.
        if self.backbone in {"lightgcn"}:
            return True
        return (
            str(self.args.get("loss_function", "")).lower() == "bpr"
            and self.backbone not in {"wmf", "neumf"}
        )

    def _edge_scores(self, edges, graph_edge_index=None):
        users = edges[0]
        items = edges[1] - self.num_users
        if self.is_gnn:
            user_emb, item_emb = self._forward_embeddings(graph_edge_index)
            return torch.sum(user_emb[users] * item_emb[items], dim=1)
        return self.model(user_indices=users, item_indices=items)

    def _positive_loss(
        self, edges, weights=None, graph_edge_index=None, neg_items=None
    ):
        """Interaction loss on an edge set.

        ``L(D; Theta)`` is used in its interaction-summed form, so ``g`` and
        ``H_inf`` share the same scale in the linear system of Eq. 13.

        ``neg_items`` optionally supplies one negative item per edge.  The
        caller reuses the same negatives for ``g`` and ``H_inf`` so that both
        are derivatives of one BPR loss realisation (Eq. 12--13), instead of
        three independently resampled losses.
        """
        if edges.numel() == 0:
            return self._zero_loss()

        users = edges[0]
        pos_items = edges[1] - self.num_users
        if self._uses_bpr_loss():
            if neg_items is None:
                neg_items = torch.randint(
                    0, self.num_items, (users.size(0),), device=self.device
                )
            else:
                neg_items = neg_items.to(self.device)
            if self.is_gnn:
                user_emb, item_emb = self._forward_embeddings(graph_edge_index)
                pos_scores = torch.sum(user_emb[users] * item_emb[pos_items], dim=1)
                neg_scores = torch.sum(user_emb[users] * item_emb[neg_items], dim=1)
            else:
                pos_scores = self.model(user_indices=users, item_indices=pos_items)
                neg_scores = self.model(user_indices=users, item_indices=neg_items)
            per_edge = -F.logsigmoid(pos_scores - neg_scores)
        else:
            scores = self._edge_scores(edges, graph_edge_index=graph_edge_index)
            per_edge = F.binary_cross_entropy_with_logits(
                scores, torch.ones_like(scores), reduction="none"
            )

        if weights is not None:
            per_edge = per_edge * weights.to(per_edge.device)
        return per_edge.sum()

    def _flat_grad(self, loss, create_graph=False):
        grads = grad(
            loss,
            self.params,
            create_graph=create_graph,
            allow_unused=True,
            retain_graph=True,
        )
        return [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, self.params)
        ]

    def _hvp(self, loss_grads, vector):
        """Hessian-vector product ``H_inf v`` via a second autograd pass."""
        dot = sum((g * v).sum() for g, v in zip(loss_grads, vector))
        hvp = grad(dot, self.params, retain_graph=True, allow_unused=True)
        return [
            h if h is not None else torch.zeros_like(p)
            for h, p in zip(hvp, self.params)
        ]

    def _conjugate_gradient(self, b, hvp_fn):
        """Solve ``(H_inf + lambda I) x = g`` with T CG iterations."""
        x = [torch.zeros_like(p) for p in self.params]
        r = [bi.detach().clone() for bi in b]
        p_vec = [ri.clone() for ri in r]
        rs_old = sum((ri * ri).sum() for ri in r)
        if float(rs_old.detach()) < 1e-20:
            return x

        for _ in range(max(self.T, 1)):
            Ap = hvp_fn(p_vec)
            pAp = sum((pi * api).sum() for pi, api in zip(p_vec, Ap))
            alpha = rs_old / (pAp + 1e-12)
            x = [xi + alpha * pi for xi, pi in zip(x, p_vec)]
            r = [ri - alpha * api for ri, api in zip(r, Ap)]
            rs_new = sum((ri * ri).sum() for ri in r)
            if float(torch.sqrt(rs_new.detach())) < 1e-8:
                break
            beta = rs_new / (rs_old + 1e-12)
            p_vec = [ri + beta * pi for ri, pi in zip(r, p_vec)]
            rs_old = rs_new
        return x

    def _add_dp_noise(self, deltas):
        """Clip ``DeltaTheta`` to norm ``C'`` and add Gaussian noise (Eq. 16).

        ``b ~ N(0, 2 C'^2 ln(1.25/delta) / epsilon^2 I)``.  This optional
        variant is disabled by default and is only used for the
        ``Unranking (+ Noise)`` ablation.
        """
        if not self.use_dp:
            return deltas

        norm = torch.sqrt(sum((d.detach() ** 2).sum() for d in deltas))
        clip = min(1.0, self.dp_clip_norm / (float(norm) + 1e-12))
        sigma = math.sqrt(
            2.0 * self.dp_clip_norm ** 2 * math.log(1.25 / self.dp_delta)
        ) / self.dp_epsilon

        noisy = []
        for d in deltas:
            noise = torch.normal(0.0, sigma, size=d.shape, device=d.device)
            noisy.append(d * clip + noise)
        logger.info(
            "DP variant: ||DeltaTheta||=%.4f, C'=%.4f, sigma=%.4f, "
            "epsilon=%.3f, delta=%.1e",
            float(norm),
            self.dp_clip_norm,
            sigma,
            self.dp_epsilon,
            self.dp_delta,
        )
        return noisy

    # ------------------------------------------------------------------
    # Entry point: Algorithm 1
    # ------------------------------------------------------------------
    def revise(self):
        logger.info("Executing Unranking for preference revision (LPF + PAPU).")
        start = time.time()
        self.model.eval()
        if hasattr(self.model, "clear_cache"):
            self.model.clear_cache()

        # Stage 1: p-hop influence scoping.
        scope_edges, target_edges, related_edges = self._p_hop_influence_scope()
        logger.info(
            "Stage 1/3 - p-hop influence scoping: |D_t|=%d, |D_inf|=%d, "
            "|D_inf \\ D_t|=%d, p=%d",
            target_edges.size(1),
            scope_edges.size(1),
            related_edges.size(1),
            self.p,
        )

        if target_edges.numel() == 0:
            elapsed = time.time() - start
            logger.warning("Target set D_t is empty; returning the model unchanged.")
            return copy.deepcopy(self.model), target_edges, elapsed

        # Split D_inf into D_t and D_inf \ D_t in the scope's own edge order,
        # so the same per-edge BPR negative can be reused by every loss term.
        scope_keys = scope_edges[0] * self.num_items + (
            scope_edges[1] - self.num_users
        )
        target_keys = target_edges[0] * self.num_items + (
            target_edges[1] - self.num_users
        )
        target_mask = torch.isin(scope_keys, target_keys)
        if int(target_mask.sum().item()) != target_edges.size(1):
            logger.warning(
                "Some target interactions are not present in the influence "
                "scope; they are excluded from g and H_inf."
            )
        target_edges_in_scope = scope_edges[:, target_mask]
        related_edges_in_scope = scope_edges[:, ~target_mask]

        # One BPR negative per scope edge, reused by g and H_inf.
        use_bpr = self._uses_bpr_loss()
        scope_neg = (
            torch.randint(
                0, self.num_items, (scope_edges.size(1),), device=self.device
            )
            if use_bpr
            else None
        )
        target_neg = scope_neg[target_mask] if scope_neg is not None else None
        related_neg = scope_neg[~target_mask] if scope_neg is not None else None

        # Stage 2: preference factor estimation.
        preference = self._compute_preference_factors(scope_edges, target_edges)
        related_weights = None
        if (
            not self.no_contextual_correction
            and related_edges_in_scope.numel() > 0
        ):
            related_weights = self._interaction_weights(
                related_edges_in_scope, preference
            )
        logger.info(
            "Stage 2/3 - preference factors: |V_inf|=%d, "
            "mean omega=%.4f, max omega=%.4f",
            len(preference),
            sum(preference.values()) / max(len(preference), 1),
            max(preference.values()) if preference else 0.0,
        )

        # Eq. (12)--(13) restrict the *interaction set* of the loss to D_t /
        # D_inf, not the GNN forward graph: Theta parameterises the full-graph
        # model M_p.  graph_edge_index=None => _forward_embeddings() uses
        # data.train_edge_index, the same graph used for evaluation.
        target_loss = self._positive_loss(
            target_edges_in_scope, neg_items=target_neg
        )
        if related_weights is None:
            total_grad_loss = target_loss
        else:
            context_loss = self._positive_loss(
                related_edges_in_scope,
                weights=related_weights,
                neg_items=related_neg,
            )
            total_grad_loss = target_loss + context_loss

        self.model.zero_grad(set_to_none=True)
        g_total = self._flat_grad(total_grad_loss, create_graph=False)

        # Hessian restricted to the D_inf interactions, H_inf.
        hessian_loss = self._positive_loss(scope_edges, neg_items=scope_neg)
        loss_grads_inf = self._flat_grad(hessian_loss, create_graph=True)

        def hvp_local(vector):
            out = [self.lam * v for v in vector]
            if loss_grads_inf is not None:
                hv = self._hvp(loss_grads_inf, vector)
                out = [a + b for a, b in zip(out, hv)]
            return out

        # Solve (H_inf + lambda I) DeltaTheta = g by CG/HVPs.
        delta = self._conjugate_gradient(g_total, hvp_local)
        delta = self._add_dp_noise(delta)

        # Theta' = Theta + DeltaTheta (Eq. 14).
        revised_model = copy.deepcopy(self.model)
        if hasattr(revised_model, "clear_cache"):
            revised_model.clear_cache()
        revised_params = [
            p for p in revised_model.parameters() if p.requires_grad
        ]
        with torch.no_grad():
            for param, update in zip(revised_params, delta):
                param.add_(update)

        delta_norm = torch.sqrt(sum((d.detach() ** 2).sum() for d in delta))
        elapsed = time.time() - start
        logger.info(
            "Stage 3/3 - PAPU: T=%d, lambda=%.4g, ||DeltaTheta||=%.4f. "
            "Revision complete in %.2fs.",
            self.T,
            self.lam,
            float(delta_norm),
            elapsed,
        )
        return revised_model, target_edges, elapsed

    # Backward-compatible entry point.
    unlearn = revise
