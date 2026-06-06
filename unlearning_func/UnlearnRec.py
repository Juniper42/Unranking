import copy
import time

import torch
import torch.nn.functional as F

from logger import Logger
from utils import determine_is_gnn

logger = Logger.get_logger("Unlearning(UnlearnRec)")


class UnlearnRec:
    def __init__(self, model, data, args, device):
        self.model = model
        self.data = data
        self.args = args
        self.device = device
        self.epochs = int(args.get("unlearnrec_epochs", 5))
        self.lr = float(args.get("unlearnrec_lr", 1e-3))
        self.batch_size = int(args.get("unlearnrec_batch_size", 4096))
        self.unlearn_wei = float(args.get("unlearnrec_unlearn_wei", 1.0))
        self.align_wei = float(args.get("unlearnrec_align_wei", 1.0))

    def _forward(self, model):
        if determine_is_gnn(self.args.get("backbone")):
            return model(edge_index=self.data.train_edge_index_after_remove)
        return model.get_embeddings()

    def _bpr(self, u_emb, pos_emb, neg_emb):
        s = (u_emb * pos_emb).sum(-1) - (u_emb * neg_emb).sum(-1)
        return -F.logsigmoid(s).mean()

    def _cal_neg_aug(self, u_emb, i_emb):
        """The `cal_neg_aug_v1` loss from the UnlearnRec reference repo.

        Penalizes the dot-product score of edges in D_t toward a lower
        bound via a softplus; equivalent to pushing those scores down.
        """
        score = (u_emb * i_emb).sum(-1)
        return F.softplus(score).mean()

    def _align(self, u_new, i_new, u_ref, i_ref, users, items):
        """L2 alignment between current and pre-unlearn predictions on the
        retained batch (`cal_positive_pred_align_v2` in the reference repo).
        """
        diff_u = (u_new[users] - u_ref[users]).pow(2).sum(-1)
        diff_i = (i_new[items] - i_ref[items]).pow(2).sum(-1)
        return (diff_u + diff_i).mean()

    def unlearn(self):
        logger.info("Executing UnlearnRec (fine-tune stage) unlearning...")
        start = time.time()

        unlearned = copy.deepcopy(self.model)
        if hasattr(unlearned, "clear_cache"):
            unlearned.clear_cache()

        # Snapshot the pre-unlearning embeddings for the alignment loss.
        with torch.no_grad():
            u_ref, i_ref = self._forward(self.model)
            u_ref, i_ref = u_ref.detach(), i_ref.detach()

        optim = torch.optim.Adam(
            [p for p in unlearned.parameters() if p.requires_grad], lr=self.lr
        )
        retained = self.data.train_edge_index_after_remove
        target = self.data.edges_to_remove

        for epoch in range(self.epochs):
            perm = torch.randperm(retained.size(1), device=self.device)
            losses = []
            for s in range(0, retained.size(1), self.batch_size):
                batch = retained[:, perm[s:s + self.batch_size]]
                u_idx = batch[0]
                pos = batch[1] - self.data.num_users
                neg = torch.randint(0, self.data.num_items, (u_idx.size(0),),
                                    device=self.device)

                u_all, i_all = self._forward(unlearned)
                # L_base: BPR on retained data.
                base = self._bpr(u_all[u_idx], i_all[pos], i_all[neg])
                # L_unlearn: push down scores on target edges.
                t_u = target[0]
                t_i = target[1] - self.data.num_users
                unlearn = self._cal_neg_aug(u_all[t_u], i_all[t_i])
                # L_align: preserve original predictions on the batch.
                align = self._align(u_all, i_all, u_ref, i_ref, u_idx, pos)

                loss = base + self.unlearn_wei * unlearn + self.align_wei * align
                optim.zero_grad()
                loss.backward()
                optim.step()
                losses.append(loss.item())
            if losses:
                logger.info(f"UnlearnRec epoch {epoch + 1}/{self.epochs} "
                            f"avg_loss={sum(losses) / len(losses):.4f}")

        elapsed = time.time() - start
        logger.info(f"UnlearnRec complete. Time: {elapsed:.2f}s.")
        return unlearned, target, elapsed
