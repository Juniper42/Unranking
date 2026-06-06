import copy
import math
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.autograd import grad

from logger import Logger
from utils import determine_is_gnn

logger = Logger.get_logger("Unlearning(Unranking)")


class Unranking:
    """
    Unranking: LPF + PAPU.

    The implementation follows the paper's Unranking framework:
    p-hop influence scoping, kernel-calibrated preference factors, and a
    preference-aware parameter update solved with CG/HVPs.
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

        def alias_value(primary, legacy, default):
            primary_value = args.get(primary, default)
            legacy_value = args.get(legacy, default)
            if primary_value == default and legacy_value != default:
                return legacy_value
            return primary_value

        # Paper hyperparameters: p-hop depth, CG iteration count T, damping.
        self.p = int(alias_value("unranking_hops", "influence_hops", 1))
        self.T = int(alias_value("unranking_cg_steps", "cg_steps", 30))
        self.lam = float(
            alias_value("unranking_damping_lambda", "damping_lambda", 1e-2)
        )
        self.max_scope_interactions = int(
            args.get("unranking_max_scope_interactions", 0)
        )
        legacy_scope_cap = int(args.get("rarl_max_scope_size", 5000))
        if self.max_scope_interactions == 0 and legacy_scope_cap != 5000:
            self.max_scope_interactions = legacy_scope_cap

        # Ablation switches matching Table 4 in the paper.
        self.no_scoping = bool(
            args.get("unranking_no_scoping", False)
            or args.get("rarl_no_localized", False)
        )
        self.no_preference_factor = bool(
            args.get("unranking_no_preference_factor", False)
            or args.get("rarl_no_salience_weight", False)
        )
        self.no_structural = bool(
            args.get("unranking_no_structural", False)
            or args.get("rarl_no_sconn", False)
        )
        self.no_semantic = bool(
            args.get("unranking_no_semantic", False) or args.get("rarl_no_sim", False)
        )
        self.no_contextual_correction = bool(
            args.get("unranking_no_contextual_correction", False)
        )
        self.no_curvature = bool(
            args.get("unranking_no_curvature", False)
            or args.get("rarl_no_curvature", False)
        )

        # Optional privacy-oriented variant.
        self.use_dp = bool(args.get("unranking_dp", False))
        self.dp_epsilon = float(args.get("unranking_dp_epsilon", 1.0))
        self.dp_delta = float(args.get("unranking_dp_delta", 1e-5))

    # ------------------------------------------------------------------
    # Edge and embedding helpers
    # ------------------------------------------------------------------
    def _empty_edges(self):
        return torch.empty((2, 0), dtype=torch.long, device=self.device)

    def _canonical_edge_tensor(self, edge_index, deduplicate=True):
        """
        Converts any user-item edge tensor into canonical forward edges
        (user, global_item_node). Reverse GNN edges are folded back.
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
            # Defensive fallback for legacy uncached tensors with local item ids.
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
        source_edges = getattr(
            self.data, "train_edge_index_original", self.data.train_edge_index
        )
        return self._canonical_edge_tensor(source_edges, deduplicate=True)

    def _target_edges(self):
        return self._canonical_edge_tensor(self.data.edges_to_remove, deduplicate=True)

    def _scope_graph(self, scope_edges):
        if scope_edges.numel() == 0:
            return self._empty_edges()
        return torch.cat([scope_edges, scope_edges.flip(0)], dim=1)

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
    # Localized Preference Factor (LPF)
    # ------------------------------------------------------------------
    def _p_hop_influence_scope(self):
        """
        Algorithm 1 in the paper: S(0)=D_t and each hop unions all
        interactions sharing the same user or item.
        """
        all_edges = self._interaction_dataset_edges()
        target_edges = self._target_edges()
        if target_edges.numel() == 0:
            return self._empty_edges(), target_edges, self._empty_edges()

        all_pairs = [
            (int(u), int(i))
            for u, i in zip(all_edges[0].cpu().tolist(), all_edges[1].cpu().tolist())
        ]
        target_pairs = {
            (int(u), int(i))
            for u, i in zip(
                target_edges[0].cpu().tolist(), target_edges[1].cpu().tolist()
            )
        }

        if self.no_scoping:
            scope_pairs = set(all_pairs) | target_pairs
        else:
            user_to_edges = defaultdict(list)
            item_to_edges = defaultdict(list)
            for pair in all_pairs:
                user_to_edges[pair[0]].append(pair)
                item_to_edges[pair[1]].append(pair)

            scope_pairs = set(target_pairs)
            frontier = set(target_pairs)
            for _ in range(max(self.p, 0)):
                expanded = set(scope_pairs)
                for u, i in frontier:
                    expanded.update(user_to_edges.get(u, []))
                    expanded.update(item_to_edges.get(i, []))
                frontier = expanded - scope_pairs
                scope_pairs = expanded
                if not frontier:
                    break

        if (
            self.max_scope_interactions > 0
            and len(scope_pairs) > self.max_scope_interactions
        ):
            ordered_scope = []
            for pair in target_pairs:
                ordered_scope.append(pair)
            for pair in all_pairs:
                if pair in scope_pairs and pair not in target_pairs:
                    ordered_scope.append(pair)
                if len(ordered_scope) >= self.max_scope_interactions:
                    break
            scope_pairs = set(ordered_scope) | target_pairs
            logger.warning(
                "Influence scope capped to %d interactions; this is a runtime "
                "safeguard and differs from the uncapped paper setting.",
                len(scope_pairs),
            )

        related_pairs = scope_pairs - target_pairs
        return (
            self._edge_tensor_from_pairs(sorted(scope_pairs)),
            target_edges,
            self._edge_tensor_from_pairs(sorted(related_pairs)),
        )

    def _kernel_calibrate(self, values):
        if not values:
            return {}
        keys = list(values.keys())
        value_tensor = torch.tensor(
            [values[k] for k in keys], dtype=torch.float, device=self.device
        )
        mu = value_tensor.mean()
        bandwidth = value_tensor.std(unbiased=False)
        if float(bandwidth.detach()) < 1e-12:
            calibrated = torch.ones_like(value_tensor)
        else:
            calibrated = torch.exp(-((value_tensor - mu) ** 2) / (2 * bandwidth ** 2))
        return {k: float(v) for k, v in zip(keys, calibrated.detach().cpu().tolist())}

    def _compute_preference_factors(self, scope_edges, target_edges):
        if scope_edges.numel() == 0:
            return {}

        scope_pairs = [
            (int(u), int(i))
            for u, i in zip(scope_edges[0].cpu().tolist(), scope_edges[1].cpu().tolist())
        ]
        target_entities = set(target_edges[0].cpu().tolist()) | set(
            target_edges[1].cpu().tolist()
        )
        entities = set()
        degrees = defaultdict(int)
        for u, i in scope_pairs:
            entities.add(u)
            entities.add(i)
            degrees[u] += 1
            degrees[i] += 1

        if self.no_preference_factor:
            return {v: 1.0 for v in entities}

        if self.no_structural and self.no_semantic:
            return {v: 1.0 for v in entities}

        max_deg = max(degrees.values()) if degrees else 1
        structural = {}
        if not self.no_structural:
            structural = {v: degrees[v] / max(max_deg, 1) for v in entities}

        semantic = {}
        if not self.no_semantic and target_entities:
            with torch.no_grad():
                user_emb, item_emb = self._forward_embeddings(self.data.train_edge_index)
                target_emb = torch.stack(
                    [
                        self._entity_embedding(int(e), user_emb, item_emb)
                        for e in sorted(target_entities)
                    ],
                    dim=0,
                )
                target_emb = F.normalize(target_emb, dim=1)
                for v in entities:
                    v_emb = self._entity_embedding(int(v), user_emb, item_emb)
                    v_emb = F.normalize(v_emb.unsqueeze(0), dim=1)
                    semantic[v] = float((v_emb @ target_emb.t()).mean().item())

        omega_struct = self._kernel_calibrate(structural) if structural else {}
        omega_sem = self._kernel_calibrate(semantic) if semantic else {}

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
        if edges.numel() == 0:
            return torch.empty(0, dtype=torch.float, device=self.device)
        if self.no_preference_factor:
            return torch.ones(edges.size(1), dtype=torch.float, device=self.device)

        weights = []
        for u, i in zip(edges[0].cpu().tolist(), edges[1].cpu().tolist()):
            weights.append(
                0.5 * (preference.get(int(u), 0.0) + preference.get(int(i), 0.0))
            )
        return torch.tensor(weights, dtype=torch.float, device=self.device)

    # ------------------------------------------------------------------
    # Preference-Aware Parameter Update (PAPU)
    # ------------------------------------------------------------------
    def _zero_loss(self):
        return sum((p.sum() * 0.0) for p in self.params)

    def _uses_bpr_loss(self):
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

    def _positive_loss(self, edges, weights=None, graph_edge_index=None):
        if edges.numel() == 0:
            return self._zero_loss()

        users = edges[0]
        pos_items = edges[1] - self.num_users
        if self._uses_bpr_loss():
            neg_items = torch.randint(
                0, self.num_items, (users.size(0),), device=self.device
            )
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
        dot = sum((g * v).sum() for g, v in zip(loss_grads, vector))
        hvp = grad(dot, self.params, retain_graph=True, allow_unused=True)
        return [
            h if h is not None else torch.zeros_like(p)
            for h, p in zip(hvp, self.params)
        ]

    def _conjugate_gradient(self, b, hvp_fn):
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
        if not self.use_dp:
            return deltas
        if self.dp_epsilon <= 0 or self.dp_delta <= 0:
            raise ValueError("DP variant requires epsilon > 0 and delta > 0.")

        dim = sum(d.numel() for d in deltas)
        sensitivity = math.sqrt(max(dim, 1))
        norm = torch.sqrt(sum((d.detach() ** 2).sum() for d in deltas))
        clip = min(1.0, sensitivity / (float(norm) + 1e-12))
        sigma = math.sqrt(2 * sensitivity ** 2 * math.log(1.25 / self.dp_delta))
        sigma /= self.dp_epsilon

        noisy = []
        for d in deltas:
            noise = torch.normal(0.0, sigma, size=d.shape, device=d.device)
            noisy.append(d * clip + noise)
        return noisy

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def unlearn(self):
        logger.info("Executing Unranking (LPF + PAPU) ...")
        start = time.time()
        self.model.eval()
        if hasattr(self.model, "clear_cache"):
            self.model.clear_cache()

        scope_edges, target_edges, related_edges = self._p_hop_influence_scope()
        logger.info(
            "LPF scope: |D_t|=%d, |D_inf|=%d, |D_inf\\D_t|=%d, p=%d",
            target_edges.size(1),
            scope_edges.size(1),
            related_edges.size(1),
            self.p,
        )

        if target_edges.numel() == 0:
            elapsed = time.time() - start
            return copy.deepcopy(self.model), target_edges, elapsed

        preference = self._compute_preference_factors(scope_edges, target_edges)
        related_weights = None
        if not self.no_contextual_correction and related_edges.numel() > 0:
            related_weights = self._interaction_weights(related_edges, preference)

        scope_graph = self._scope_graph(scope_edges)

        target_loss = self._positive_loss(
            target_edges, graph_edge_index=scope_graph
        )
        if related_weights is None:
            total_grad_loss = target_loss
        else:
            context_loss = self._positive_loss(
                related_edges,
                weights=related_weights,
                graph_edge_index=scope_graph,
            )
            total_grad_loss = target_loss + context_loss

        self.model.zero_grad(set_to_none=True)
        g_total = self._flat_grad(total_grad_loss, create_graph=False)

        if self.no_curvature:
            loss_grads_inf = None
        else:
            hessian_loss = self._positive_loss(scope_edges, graph_edge_index=scope_graph)
            loss_grads_inf = self._flat_grad(hessian_loss, create_graph=True)

        def hvp_local(vector):
            out = [self.lam * v for v in vector]
            if loss_grads_inf is not None:
                hv = self._hvp(loss_grads_inf, vector)
                out = [a + b for a, b in zip(out, hv)]
            return out

        delta = self._conjugate_gradient(g_total, hvp_local)
        delta = self._add_dp_noise(delta)

        unlearned_model = copy.deepcopy(self.model)
        if hasattr(unlearned_model, "clear_cache"):
            unlearned_model.clear_cache()
        with torch.no_grad():
            for param, update in zip(unlearned_model.parameters(), delta):
                if param.requires_grad:
                    param.add_(update)

        elapsed = time.time() - start
        logger.info("Unranking complete. Time: %.2fs.", elapsed)
        return unlearned_model, target_edges, elapsed


# Backward-compatible class alias for older scripts.
RARU = Unranking
