import copy
import time

import torch
import torch.nn.functional as F
from torch.autograd import grad

from logger import Logger
from utils import determine_is_gnn

logger = Logger.get_logger("Unlearning(RRL)")


class RRL:
    def __init__(self, model, data, args, device):
        self.model = model
        self.data = data
        self.args = args
        self.device = device
        self.epochs = int(args.get("rrl_epochs", 5))
        self.lr = float(args.get("rrl_lr", 1e-3))
        self.batch_size = int(args.get("rrl_batch_size", 4096))
        self.fisher_eps = float(args.get("rrl_fisher_eps", 1e-4))
        self.fisher_samples = int(args.get("rrl_fisher_samples", 4))

    def _bpr_loss(self, model, edges, sign=+1.0):
        users = edges[0]
        pos = edges[1] - self.data.num_users
        neg = torch.randint(0, self.data.num_items, (users.size(0),),
                            device=self.device)
        if determine_is_gnn(self.args.get("backbone")):
            u_all, i_all = model(
                edge_index=self.data.train_edge_index_after_remove
            )
        else:
            u_all, i_all = model.get_embeddings()
        diff = (u_all[users] * i_all[pos]).sum(-1) \
               - (u_all[users] * i_all[neg]).sum(-1)
        # sign = +1 -> standard BPR, sign = -1 -> reverse BPR (RBPR).
        return sign * -F.logsigmoid(sign * diff).mean()

    def _estimate_fisher(self, model):
        """Diagonal FIM estimate via squared gradients on retained data."""
        params = [p for p in model.parameters() if p.requires_grad]
        fisher = [torch.zeros_like(p) for p in params]
        retained = self.data.train_edge_index_after_remove
        n = retained.size(1)
        sample_size = min(self.fisher_samples * self.batch_size, n)
        for _ in range(self.fisher_samples):
            idx = torch.randint(0, n, (min(self.batch_size, n),),
                                device=self.device)
            loss = self._bpr_loss(model, retained[:, idx], sign=+1.0)
            grads = grad(loss, params, allow_unused=True, retain_graph=False)
            for f, g in zip(fisher, grads):
                if g is not None:
                    f.add_(g.detach().pow(2))
        return [f / max(self.fisher_samples, 1) for f in fisher]

    def unlearn(self):
        logger.info("Executing RRL (Reverse Learning) unlearning...")
        start = time.time()

        unlearned = copy.deepcopy(self.model)
        if hasattr(unlearned, "clear_cache"):
            unlearned.clear_cache()
        params = [p for p in unlearned.parameters() if p.requires_grad]

        # Fisher Information Matrix diagonal on the retained data.
        fisher = self._estimate_fisher(unlearned)
        scale = [1.0 / (f + self.fisher_eps) for f in fisher]

        # Reverse-BPR fine-tuning, manually applying FIM-scaled gradients.
        target = self.data.edges_to_remove
        if target.numel() == 0:
            elapsed = time.time() - start
            return unlearned, target, elapsed

        for epoch in range(self.epochs):
            perm = torch.randperm(target.size(1), device=self.device)
            losses = []
            for s in range(0, target.size(1), self.batch_size):
                batch = target[:, perm[s:s + self.batch_size]]
                # Standard BPR with negated sign = reverse BPR.
                loss = self._bpr_loss(unlearned, batch, sign=-1.0)
                grads = grad(loss, params, allow_unused=True)
                with torch.no_grad():
                    for p, g, sc in zip(params, grads, scale):
                        if g is None:
                            continue
                        p.add_(-self.lr * (sc * g))
                losses.append(loss.item())
            if losses:
                logger.info(f"RRL epoch {epoch + 1}/{self.epochs} "
                            f"avg_loss={sum(losses) / len(losses):.4f}")

        elapsed = time.time() - start
        logger.info(f"RRL complete. Time: {elapsed:.2f}s.")
        return unlearned, target, elapsed
