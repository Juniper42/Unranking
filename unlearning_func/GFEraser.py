import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import Logger
from utils import determine_is_gnn

logger = Logger.get_logger("Unlearning(GFEraser)")


def _info_nce(view1, view2, temperature=1.0):
    v1, v2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos = (v1 * v2).sum(dim=-1)
    pos = torch.exp(pos / temperature)
    ttl = torch.exp(v1 @ v2.t() / temperature).sum(dim=1)
    return -torch.log(pos / (ttl + 1e-8) + 1e-8).mean()


def _js_divergence(e1, e2):
    p1 = F.softmax(e1, dim=1)
    p2 = F.softmax(e2, dim=1)
    avg = 0.5 * (p1 + p2)
    return 0.5 * (F.kl_div(p1.log(), avg, reduction="batchmean")
                  + F.kl_div(p2.log(), avg, reduction="batchmean"))


class _MLPFilter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 2 * dim)
        self.fc2 = nn.Linear(2 * dim, 2 * dim)
        self.fc3 = nn.Linear(2 * dim, dim)
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_normal_(m.weight)

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        x = self._gelu(self.fc1(x))
        x = self._gelu(self.fc2(x))
        x = self.fc3(x)
        return F.normalize(x, p=2, dim=-1)


class _AttentionFuse(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(dim, dim), nn.Tanh(),
            nn.Linear(dim, 1, bias=False),
        )

    def forward(self, e1, e2):
        att = torch.cat([self.query(e1), self.query(e2)], dim=-1)
        w = F.softmax(att, dim=-1)
        return w[:, 0:1] * e1 + w[:, 1:2] * e2


class _PreFilter(nn.Module):
    def __init__(self, user_emb, item_emb):
        super().__init__()
        dim = user_emb.size(1)
        # Pretrained embeddings - kept trainable to mimic the reference repo.
        self.user_embedding = nn.Embedding.from_pretrained(user_emb.clone(),
                                                           freeze=False)
        self.item_embedding = nn.Embedding.from_pretrained(item_emb.clone(),
                                                           freeze=False)
        # Negative-graph embeddings that capture the outdated preferences.
        self.neg_user_embedding = nn.Embedding(user_emb.size(0), dim)
        self.neg_item_embedding = nn.Embedding(item_emb.size(0), dim)
        nn.init.xavier_uniform_(self.neg_user_embedding.weight)
        nn.init.xavier_uniform_(self.neg_item_embedding.weight)

        self.mlp_u = _MLPFilter(dim)
        self.mlp_i = _MLPFilter(dim)
        self.att_u = _AttentionFuse(dim)
        self.att_i = _AttentionFuse(dim)

    def predict_embeddings(self):
        fu = self.mlp_u(self.user_embedding.weight)
        fi = self.mlp_i(self.item_embedding.weight)
        u = self.att_u(self.user_embedding.weight, fu)
        i = self.att_i(self.item_embedding.weight, fi)
        return u, i


class GFEraser:
    def __init__(self, model, data, args, device):
        self.model = model
        self.data = data
        self.args = args
        self.device = device
        self.epochs = int(args.get("gferaser_epochs", 20))
        self.lr = float(args.get("gferaser_lr", 1e-3))
        self.batch_size = int(args.get("gferaser_batch_size", 1024))
        self.cl_weight = float(args.get("gferaser_cl_weight", 0.4))
        self.pos_bpr_weight = float(args.get("gferaser_pos_bpr_weight", 1e-3))
        self.temp = float(args.get("gferaser_temp", 1.0))
        self.reg_weight = float(args.get("gferaser_reg_weight", 1e-4))

    def _get_pretrained_embeddings(self):
        if determine_is_gnn(self.args.get("backbone")):
            with torch.no_grad():
                u, i = self.model(edge_index=self.data.train_edge_index_after_remove)
            return u.detach(), i.detach()
        with torch.no_grad():
            u, i = self.model.get_embeddings()
        return u.detach(), i.detach()

    def _bpr(self, u, pos, neg):
        s_pos = (u * pos).sum(-1)
        s_neg = (u * neg).sum(-1)
        return -F.logsigmoid(s_pos - s_neg).mean()

    def _intra_user_negatives(self, users):
        """Sample one negative item per user from the global pool.

        The reference repo samples from the user's already-interacted items;
        we approximate with global random sampling to keep the dependency
        surface small.
        """
        return torch.randint(0, self.data.num_items,
                             (users.size(0),), device=self.device)

    def _sample_batches(self, neg_edges):
        n = neg_edges.size(1)
        perm = torch.randperm(n, device=self.device)
        for s in range(0, n, self.batch_size):
            idx = perm[s:s + self.batch_size]
            yield neg_edges[:, idx]

    def unlearn(self):
        logger.info("Executing GFEraser unlearning...")
        start = time.time()

        u_emb, i_emb = self._get_pretrained_embeddings()
        pre = _PreFilter(u_emb, i_emb).to(self.device)
        optim = torch.optim.Adam(pre.parameters(), lr=self.lr)

        neg_edges = self.data.edges_to_remove  # interactions to forget
        pos_edges = self.data.train_edge_index_after_remove

        for epoch in range(self.epochs):
            losses = []
            for batch in self._sample_batches(neg_edges):
                if batch.size(1) == 0:
                    continue
                users = batch[0]
                pos_items = batch[1] - self.data.num_users
                neg_items = self._intra_user_negatives(users)

                # Negative-graph BPR (learns outdated preferences).
                ne_u = pre.neg_user_embedding(users)
                ne_pi = pre.neg_item_embedding(pos_items)
                ne_ni = pre.neg_item_embedding(neg_items)
                neg_bpr = self._bpr(ne_u, ne_pi, ne_ni)

                # Filtered original embeddings.
                ou_e = pre.user_embedding(users)
                opi_e = pre.item_embedding(pos_items)
                oni_e = pre.item_embedding(neg_items)
                fu = pre.mlp_u(ou_e)
                fi = pre.mlp_i(opi_e)

                # Maximize JS divergence between filtered and neg-graph.
                js_loss = -(_js_divergence(fu, ne_u.detach())
                            + _js_divergence(fi, ne_pi.detach()))

                # Contrastive loss preserves invariant preference signals.
                cl = _info_nce(ou_e, fu, self.temp) + _info_nce(opi_e, fi, self.temp)

                # Final attention-fused BPR (downstream utility).
                final_u = pre.att_u(ou_e, fu)
                final_pi = pre.att_i(opi_e, fi)
                pos_bpr = self._bpr(final_u, final_pi, oni_e)

                reg = (ou_e.pow(2).sum() + opi_e.pow(2).sum() + oni_e.pow(2).sum())

                loss = (neg_bpr
                        + self.cl_weight * cl
                        + self.pos_bpr_weight * pos_bpr
                        + js_loss
                        + self.reg_weight * reg)
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())
            if losses:
                logger.info(f"GFEraser epoch {epoch + 1}/{self.epochs} "
                            f"avg_loss={sum(losses) / len(losses):.4f}")

        # Write filtered embeddings back into a fresh copy of the model.
        unlearned = copy.deepcopy(self.model)
        if hasattr(unlearned, "clear_cache"):
            unlearned.clear_cache()
        with torch.no_grad():
            u_final, i_final = pre.predict_embeddings()
            if hasattr(unlearned, "user_emb"):
                unlearned.user_emb.weight.data.copy_(u_final.detach())
            if hasattr(unlearned, "item_emb"):
                unlearned.item_emb.weight.data.copy_(i_final.detach())

        elapsed = time.time() - start
        logger.info(f"GFEraser complete. Time: {elapsed:.2f}s.")
        return unlearned, self.data.edges_to_remove, elapsed
