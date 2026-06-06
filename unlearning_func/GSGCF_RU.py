import copy
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.utils import k_hop_subgraph, negative_sampling, to_undirected

from logger import Logger

logger = Logger.get_logger("Unlearning(GSGCF-RU)")


class GSGCF_RU:
    """
    Implementation of GSGCF-RU (General Strategy Graph Collaborative Filtering for Recommendation Unlearning).
    This method is based on GNNDelete and achieves efficient unlearning for graph-based models
    by designing a Learnable Deletion Operator (LDO) and two consistency constraints:
    1. Unlearning Edge Consistency (UEC)
    2. Feature Representation Consistency (FRC)
    """

    def __init__(self, model, data, args, device):
        """
        Initializes the GSGCF-RU unlearner.

        Args:
            model: The graph collaborative filtering model to perform unlearning on.
            data: The data object.
            args: A dictionary of configuration arguments.
            device: The computation device.
        """
        self.model = model
        self.data = data
        self.args = args
        self.device = device

        # GSGCF-RU hyperparameters
        self.epochs = args.get("unlearning_epochs", 100)
        self.lr = args.get("unlearning_lr", 0.001)
        self.lambda_weight = args.get("lambda", 0.7)  # Weight for UEC vs. FRC loss
        self.neg_samples = args.get("unlearning_neg_samples", 10)
        self.influence_hops = args.get("influence_hops", 2)
        self.ldo_alpha = args.get("ldo_alpha", 0.1)  # Residual coefficient for LDO
        self.ldo_reg_weight = args.get(
            "ldo_reg_weight", 1e-5
        )  # L2 regularization for LDO

        self._initialize_ldo_layers()

    def _initialize_ldo_layers(self):
        """Creates the Learnable Deletion Operator (LDO) layers."""
        emb_dim = self.args.get("emb_dim", 64)
        num_layers = self.args.get("num_layers", 3)

        # Design different LDO architectures for different model layers
        ldo_layers_list = []
        for l in range(num_layers):
            if l == 0:  # Wider MLP for early layers to capture more feature changes
                layer = nn.Sequential(
                    nn.Linear(emb_dim, emb_dim * 2),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.2),
                    nn.Linear(emb_dim * 2, emb_dim),
                )
            elif l < num_layers - 1:  # Standard MLP for intermediate layers
                layer = nn.Sequential(
                    nn.Linear(emb_dim, emb_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(emb_dim, emb_dim),
                )
            else:  # Smaller perturbation for the final layer
                layer = nn.Sequential(
                    nn.Linear(emb_dim, emb_dim // 2),
                    nn.LeakyReLU(0.2),
                    nn.Linear(emb_dim // 2, emb_dim),
                )
            ldo_layers_list.append(layer)

        self.ldo_layers = nn.ModuleList(ldo_layers_list).to(self.device)
        self.subgraph_node_mask = None
        logger.info(
            f"Created {len(self.ldo_layers)} Learnable Deletion Operator layers."
        )

    def prepare_subgraph_and_masks(self, edges_to_remove):
        """
        Prepares the local subgraph and masks needed for training the LDO.
        The subgraph contains nodes and edges within a k-hop vicinity of the nodes
        involved in the edges to be unlearned.
        """
        logger.info("Preparing local subgraph and masks...")

        # Identify nodes involved in the unlearning task
        deleted_nodes = torch.unique(edges_to_remove)

        # Extract the k-hop subgraph around the deleted nodes
        edge_index = self.data.train_edge_index
        try:
            _, _, _, mask = k_hop_subgraph(
                deleted_nodes,
                self.influence_hops,
                edge_index,
                relabel_nodes=False,
                num_nodes=self.data.num_users + self.data.num_items,
            )
            self.subgraph_edge_mask = mask
            subgraph_nodes = torch.unique(edge_index[:, mask])
        except Exception as e:
            logger.warning(f"k_hop_subgraph failed: {e}. Using a simpler fallback.")
            # Fallback: Use edges directly connected to the deleted nodes
            mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=self.device)
            for i in range(edge_index.size(1)):
                if (
                    edge_index[0, i] in deleted_nodes
                    or edge_index[1, i] in deleted_nodes
                ):
                    mask[i] = True
            self.subgraph_edge_mask = mask
            subgraph_nodes = deleted_nodes

        # Create a boolean mask for nodes in the subgraph
        self.subgraph_node_mask = torch.zeros(
            self.data.num_users + self.data.num_items,
            dtype=torch.bool,
            device=self.device,
        )
        self.subgraph_node_mask[subgraph_nodes] = True

        logger.info(
            f"Local subgraph contains {self.subgraph_node_mask.sum().item()} nodes and {self.subgraph_edge_mask.sum().item()} edges."
        )
        return self.subgraph_node_mask

    def compute_loss(self, model_output, edges_to_remove, original_embeddings):
        """
        Computes the GSGCF-RU loss function, which includes:
        1. Unlearning Edge Consistency (UEC) Loss
        2. Feature Representation Consistency (FRC) Loss
        3. L2 regularization on LDO parameters
        """
        # 1. Unlearning Edge Consistency (UEC) Loss:
        # Aims to make the prediction scores of deleted edges similar to those of random negative edges.
        neg_edge_index = negative_sampling(
            edge_index=self.data.train_edge_index,
            num_nodes=self.data.num_users + self.data.num_items,
            num_neg_samples=edges_to_remove.size(1) * self.neg_samples,
        )

        pos_logits = (
            model_output[edges_to_remove[0]] * model_output[edges_to_remove[1]]
        ).sum(dim=1)
        neg_logits = (
            model_output[neg_edge_index[0]] * model_output[neg_edge_index[1]]
        ).sum(dim=1)
        neg_logits_avg = neg_logits.view(-1, self.neg_samples).mean(dim=1)
        loss_uec = F.mse_loss(pos_logits, neg_logits_avg)

        # 2. Feature Representation Consistency (FRC) Loss:
        # Aims to preserve the prediction scores for edges in the local subgraph that were not deleted.
        local_mask = self.subgraph_edge_mask.clone()
        # Exclude the edges to be removed from the FRC calculation
        edge_map = {
            tuple(e.tolist()): i for i, e in enumerate(self.data.train_edge_index.t())
        }
        for i in range(edges_to_remove.size(1)):
            edge_tuple = tuple(edges_to_remove[:, i].tolist())
            if edge_tuple in edge_map:
                local_mask[edge_map[edge_tuple]] = False

        if local_mask.sum() > 0:
            local_edges = self.data.train_edge_index[:, local_mask]
            new_logits = (
                model_output[local_edges[0]] * model_output[local_edges[1]]
            ).sum(dim=1)
            orig_logits = (
                original_embeddings[local_edges[0]]
                * original_embeddings[local_edges[1]]
            ).sum(dim=1)
            loss_frc = F.mse_loss(new_logits, orig_logits)
        else:
            loss_frc = torch.tensor(0.0, device=self.device)

        # 3. L2 Regularization for LDO
        ldo_reg = sum((p**2).sum() for p in self.ldo_layers.parameters())
        total_loss = (
            self.lambda_weight * loss_uec
            + (1 - self.lambda_weight) * loss_frc
            + self.ldo_reg_weight * ldo_reg
        )

        return total_loss, loss_uec, loss_frc

    def apply_ldo(self, embeddings):
        """
        Applies the Learnable Deletion Operator (LDO) to node embeddings
        using a residual connection controlled by alpha.
        """
        updated_embeddings = embeddings.clone()
        if self.subgraph_node_mask is not None and self.subgraph_node_mask.sum() > 0:
            nodes_to_update = torch.where(self.subgraph_node_mask)[0]
            current_embs = updated_embeddings[nodes_to_update]

            for layer in self.ldo_layers:
                update = layer(current_embs)
                current_embs = current_embs + self.ldo_alpha * update

            updated_embeddings[nodes_to_update] = current_embs
        return updated_embeddings

    def train_ldo(self, edges_to_remove):
        """
        Trains the Learnable Deletion Operator (LDO).

        Args:
            edges_to_remove: The edges to be unlearned.
        """
        logger.info("Training the Learnable Deletion Operator (LDO)...")

        with torch.no_grad():
            original_user_emb, original_item_emb = self.model.get_embeddings()
            original_embeddings = torch.cat(
                [original_user_emb, original_item_emb], dim=0
            ).to(self.device)

        optimizer = torch.optim.Adam(self.ldo_layers.parameters(), lr=self.lr)

        best_loss = float("inf")
        best_state_dict = None

        for epoch in range(self.epochs):
            self.ldo_layers.train()
            updated_embeddings = self.apply_ldo(original_embeddings)
            loss, loss_uec, loss_frc = self.compute_loss(
                updated_embeddings, edges_to_remove, original_embeddings
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs}: Loss={loss.item():.4f}, UEC={loss_uec.item():.4f}, FRC={loss_frc.item():.4f}"
                )

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state_dict = {
                    k: v.clone() for k, v in self.ldo_layers.state_dict().items()
                }

        logger.info(f"LDO training finished. Best loss: {best_loss:.4f}")
        if best_state_dict:
            self.ldo_layers.load_state_dict(best_state_dict)

    def _create_unlearned_model(self):
        """
        Creates the final unlearned model by integrating the trained LDO.
        This is done by dynamically modifying the `get_embeddings` method of a model copy.
        """
        unlearned_model = copy.deepcopy(self.model)
        unlearned_model.ldo_layers = self.ldo_layers
        unlearned_model.subgraph_node_mask = self.subgraph_node_mask
        unlearned_model.ldo_alpha = self.ldo_alpha

        original_get_embeddings = unlearned_model.get_embeddings

        # Define the new get_embeddings method that applies the LDO
        def new_get_embeddings(self_model):
            # First, get the embeddings from the original GNN layers
            user_emb, item_emb = original_get_embeddings()
            all_emb = torch.cat([user_emb, item_emb], dim=0)

            # Apply the trained LDO to the embeddings
            updated_emb = self_model.apply_ldo(all_emb)

            # Split back into user and item embeddings
            updated_user_emb, updated_item_emb = torch.split(
                updated_emb, [self_model.data.num_users, self_model.data.num_items]
            )
            return updated_user_emb, updated_item_emb

        def apply_ldo_method(self_model, embeddings):
            updated_embeddings = embeddings.clone()
            if (
                self_model.subgraph_node_mask is not None
                and self_model.subgraph_node_mask.sum() > 0
            ):
                nodes_to_update = torch.where(self_model.subgraph_node_mask)[0]
                current_embs = updated_embeddings[nodes_to_update]
                for layer in self_model.ldo_layers:
                    update = layer(current_embs)
                    current_embs = current_embs + self_model.ldo_alpha * update
                updated_embeddings[nodes_to_update] = current_embs
            return updated_embeddings

        # Monkey-patch the methods onto the new model instance
        unlearned_model.apply_ldo = types.MethodType(apply_ldo_method, unlearned_model)
        unlearned_model.get_embeddings = types.MethodType(
            new_get_embeddings, unlearned_model
        )

        return unlearned_model

    def unlearn(self):
        """
        Executes the GSGCF-RU unlearning process.

        Returns:
            A tuple of (unlearned_model, edges_to_remove, unlearning_time).
        """
        logger.info("Executing GSGCF-RU unlearning process...")
        start_time = time.time()

        edges_to_remove = self.data.edges_to_remove

        # 1. Prepare subgraph around the affected nodes
        self.prepare_subgraph_and_masks(edges_to_remove)

        # 2. Train the Learnable Deletion Operator
        self.train_ldo(edges_to_remove)

        # 3. Create the final model with the integrated LDO
        unlearned_model = self._create_unlearned_model()

        unlearning_time = time.time() - start_time
        logger.info(
            f"GSGCF-RU unlearning complete. Time taken: {unlearning_time:.2f} seconds."
        )

        return unlearned_model, edges_to_remove, unlearning_time
