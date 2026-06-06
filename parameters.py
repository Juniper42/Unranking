import argparse


def str2bool(v):
    """Converts a string to a boolean value."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class Config:
    """Configuration class for managing command-line arguments."""

    @staticmethod
    def get_config():
        """Gets the command-line argument configuration."""
        parser = argparse.ArgumentParser(
            description="Comparative Experiments of Unlearning Methods in Recommender Systems"
        )

        # Core Parameters
        parser.add_argument(
            "--method",
            type=str,
            default="retrain",
            choices=[
                "retrain",
                "gsgcf-ru",
                "receraser",
                "unranking",
                "raru",
                "ifru",
                "sisa",
                "certified_removal",
                "gferaser",
                "ultrare",
                "rrl",
                "unlearnrec",
            ],
            help="Unlearning method to use.",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            default="ml-1m",
            choices=["ml-1m", "yelp2018", "amazon-book", "ml-10m", "netflix"],
            help="Dataset to use.",
        )
        parser.add_argument(
            "--backbone",
            type=str,
            default="lightgcn",
            choices=[
                "lightgcn",
                "gat",
                "graphsage",
                "mlp",
                "gmf",
                "neumf",
                "wmf",
                "gcn",
                "gin",
            ],
            help="Backbone model for recommendations.",
        )
        parser.add_argument(
            "--is_attack",
            type=str2bool,
            default=True,
            help="Whether to perform a Membership Inference Attack.",
        )
        parser.add_argument(
            "--evaluate_forgetting",
            type=str2bool,
            default=True,
            help="Whether to evaluate the forgetting effect.",
        )
        parser.add_argument(
            "--split_method",
            type=str,
            default="temporal",
            help="Dataset splitting method (e.g., temporal).",
        )
        parser.add_argument(
            "--k", type=int, default=10, help="Size of the recommendation list (top-k)."
        )
        parser.add_argument(
            "--use_cache",
            type=str2bool,
            default=True,
            help="Whether to use cached data.",
        )
        parser.add_argument(
            "--use_model_cache",
            type=str2bool,
            default=True,
            help="Whether to use cached model weights.",
        )
        parser.add_argument(
            "--output_result_path",
            type=str,
            default="",
            help="Path to save the experiment results in a CSV file.",
        )
        parser.add_argument(
            "--num_runs",
            type=int,
            default=10,
            help="Number of experimental runs for averaging results.",
        )

        # Training Parameters
        parser.add_argument(
            "--lr", type=float, default=1e-3, help="Learning rate for the main model."
        )
        parser.add_argument(
            "--epoch", type=int, default=100, help="Number of training epochs."
        )
        parser.add_argument(
            "--batch_size", type=int, default=1024, help="Training batch size."
        )
        parser.add_argument(
            "--emb_dim", type=int, default=64, help="Embedding dimension."
        )
        parser.add_argument(
            "--num_layers", type=int, default=3, help="Number of GNN layers."
        )
        parser.add_argument(
            "--dropout_rate", type=float, default=0.2, help="Dropout rate."
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-4,
            help="Weight decay (L2 regularization).",
        )
        parser.add_argument(
            "--test_size",
            type=float,
            default=0.1,
            help="Proportion of the dataset to use for testing.",
        )
        parser.add_argument(
            "--val_size",
            type=float,
            default=0.1,
            help="Proportion for validation set. If 0, test set is used for validation.",
        )
        parser.add_argument(
            "--eval_interval",
            type=int,
            default=5,
            help="Interval (in epochs) for evaluation.",
        )
        parser.add_argument(
            "--early_stop",
            type=str2bool,
            default=True,
            help="Whether to enable early stopping.",
        )
        parser.add_argument(
            "--patience",
            type=int,
            default=5,
            help="Patience for early stopping.",
        )
        parser.add_argument(
            "--early_stop_metric",
            type=str,
            default="recall@k",
            choices=["recall@k", "ndcg@k"],
            help="Metric for early stopping.",
        )
        parser.add_argument(
            "--scheduler_type",
            type=str,
            default="cosine",
            choices=["cosine", "plateau", "exponential", "default"],
            help="Learning rate scheduler type.",
        )
        parser.add_argument(
            "--lr_gamma",
            type=float,
            default=0.95,
            help="Decay factor for the exponential learning rate scheduler.",
        )
        parser.add_argument(
            "--neg_samples",
            type=int,
            default=20,
            help="Number of negative samples per positive sample.",
        )
        parser.add_argument(
            "--reg_weight",
            type=float,
            default=1e-3,
            help="Weight for L2 regularization loss.",
        )
        parser.add_argument(
            "--margin",
            type=float,
            default=0.0,
            help="Margin for BPR loss.",
        )
        parser.add_argument(
            "--loss_function",
            type=str,
            default="bpr",
            choices=["bpr", "pointwise_bce", "infonce"],
            help="Loss function type.",
        )
        parser.add_argument(
            "--loss_temperature",
            type=float,
            default=0.2,
            help="Temperature parameter for InfoNCE loss.",
        )

        # MIA Parameters
        parser.add_argument(
            "--attack_batch_size",
            type=int,
            default=512,
            help="Batch size for the attack model.",
        )
        parser.add_argument(
            "--attack_epochs",
            type=int,
            default=100,
            help="Number of training epochs for the attack model.",
        )
        parser.add_argument(
            "--attack_lr",
            type=float,
            default=0.001,
            help="Learning rate for the attack model.",
        )

        # Unlearning Parameters
        parser.add_argument(
            "--unlearning_ratio",
            type=float,
            default=0.1,
            help="Proportion of users or items to be unlearned.",
        )
        parser.add_argument(
            "--unlearning_task",
            type=str,
            default="interaction",
            choices=["interaction", "item"],
            help="Type of unlearning task: 'interaction' or 'item'.",
        )
        parser.add_argument(
            "--unlearning_interaction_ratio",
            type=float,
            default=0.2,
            help="Proportion of a user's interactions to remove in 'interaction' unlearning.",
        )

        # General parameters that might be used by multiple methods
        parser.add_argument(
            "--influence_hops",
            type=int,
            default=1,
            help="Number of hops for influence calculation.",
        )
        parser.add_argument(
            "--iteration",
            type=int,
            default=10,
            help="Number of iterations for influence approximation.",
        )
        parser.add_argument(
            "--damp",
            type=float,
            default=0.0001,
            help="Damping factor for influence calculation.",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=0.1,
            help="Scaling factor for influence calculation.",
        )
        parser.add_argument(
            "--use_high_impact",
            type=str2bool,
            default=True,
            help="Whether to use high-impact edges for unlearning (e.g., in GIF-like methods).",
        )
        parser.add_argument(
            "--degree_weight",
            type=float,
            default=0.5,
            help="Weight for node degree in impact scoring.",
        )
        parser.add_argument(
            "--score_weight",
            type=float,
            default=0.5,
            help="Weight for interaction score in impact scoring.",
        )
        parser.add_argument(
            "--high_impact_ratio",
            type=float,
            default=1.0,
            help="Ratio of high-impact edges to consider.",
        )

        # GSGCF-RU Parameters
        parser.add_argument(
            "--unlearning_epochs",
            type=int,
            default=100,
            help="Number of epochs for GSGCF-RU fine-tuning.",
        )
        parser.add_argument(
            "--unlearning_lr",
            type=float,
            default=1e-3,
            help="Learning rate for GSGCF-RU fine-tuning.",
        )
        parser.add_argument(
            "--lambda",
            type=float,
            default=0.5,
            help="Weight to balance consistency and causality in GSGCF-RU.",
        )
        parser.add_argument(
            "--unlearning_neg_samples",
            type=int,
            default=5,
            help="Number of negative samples for GSGCF-RU.",
        )

        # RecEraser Parameters
        parser.add_argument(
            "--attention_size",
            type=int,
            default=None,
            help="Attention mechanism dimension for RecEraser.",
        )
        parser.add_argument(
            "--num_partitions",
            type=int,
            default=3,
            help="Number of partitions for RecEraser.",
        )
        parser.add_argument(
            "--partition_type",
            type=str,
            default="interaction",
            choices=["user", "item", "interaction"],
            help="Partitioning strategy for RecEraser.",
        )
        parser.add_argument(
            "--sub_epochs",
            type=int,
            default=10,
            help="Number of training epochs for sub-models in RecEraser.",
        )
        parser.add_argument(
            "--agg_epochs",
            type=int,
            default=5,
            help="Number of aggregation epochs in RecEraser.",
        )
        parser.add_argument(
            "--sub_lr",
            type=float,
            default=0.001,
            help="Learning rate for sub-models in RecEraser.",
        )
        parser.add_argument(
            "--att_lr",
            type=float,
            default=0.001,
            help="Learning rate for the attention mechanism in RecEraser.",
        )

        # IFRU Parameters
        parser.add_argument(
            "--pruning_rates",
            nargs="+",
            type=float,
            default=[1.0, 0.5, 0.25],
            help="Pruning rates for IFRU, e.g., --pruning_rates 0.8 0.6 0.4",
        )

        # SISA Parameters
        parser.add_argument(
            "--sisa_num_shards",
            type=int,
            default=5,
            help="Number of data shards for SISA.",
        )
        parser.add_argument(
            "--sisa_num_slices",
            type=int,
            default=1,
            help="Number of data slices for SISA, supporting incremental training.",
        )
        parser.add_argument(
            "--sisa_aggregation",
            type=str,
            default="uniform",
            choices=["uniform", "weighted"],
            help="Aggregation strategy for SISA: 'uniform' or 'weighted'.",
        )

        # CertifiedRemoval Parameters
        parser.add_argument(
            "--certified_lam",
            type=float,
            default=1e-4,
            help="L2 regularization parameter for CertifiedRemoval.",
        )
        parser.add_argument(
            "--certified_std",
            type=float,
            default=10.0,
            help="Target perturbation standard deviation for CertifiedRemoval.",
        )
        parser.add_argument(
            "--certified_num_steps",
            type=int,
            default=100,
            help="Number of LBFGS optimization steps for CertifiedRemoval.",
        )
        parser.add_argument(
            "--certified_subsample_ratio",
            type=float,
            default=1.0,
            help="Negative sampling ratio for CertifiedRemoval.",
        )
        parser.add_argument(
            "--certified_batch_size_hessian",
            type=int,
            default=50000,
            help="Batch size for Hessian matrix computation in CertifiedRemoval.",
        )
        parser.add_argument(
            "--certified_finetune_lr",
            type=float,
            default=1e-4,
            help="Learning rate for the fine-tuning step in CertifiedRemoval.",
        )
        parser.add_argument(
            "--certified_finetune_epochs",
            type=int,
            default=5,
            help="Number of epochs for the fine-tuning step in CertifiedRemoval.",
        )
        parser.add_argument(
            "--certified_train_mode",
            type=str,
            default="ovr",
            choices=["ovr", "binary"],
            help="Training mode for CertifiedRemoval: 'ovr' (One-vs-Rest) or 'binary'.",
        )

        # Unranking Parameters (LPF + PAPU; paper Section 4 / Section 5.1.6).
        parser.add_argument(
            "--unranking_hops",
            type=int,
            default=1,
            help="p-hop influence scope depth for LPF. Search in {0,1,2,3,4}.",
        )
        parser.add_argument(
            "--unranking_damping_lambda",
            type=float,
            default=1e-2,
            help="Damping coefficient lambda in (H_inf + lambda I) Delta = g.",
        )
        parser.add_argument(
            "--unranking_cg_steps",
            type=int,
            default=30,
            help="Conjugate-gradient iteration count T. Search in {10,20,30,40,50}.",
        )
        parser.add_argument(
            "--unranking_max_scope_interactions",
            type=int,
            default=0,
            help="Optional runtime cap for |D_inf|. 0 disables the cap, matching the paper.",
        )
        parser.add_argument(
            "--unranking_no_scoping", type=str2bool, default=False,
            help="Ablation: use the full interaction dataset instead of D_inf.",
        )
        parser.add_argument(
            "--unranking_no_preference_factor", type=str2bool, default=False,
            help="Ablation: use uniform contextual weights.",
        )
        parser.add_argument(
            "--unranking_no_structural", type=str2bool, default=False,
            help="Ablation: drop I_struct from the preference factor.",
        )
        parser.add_argument(
            "--unranking_no_semantic", type=str2bool, default=False,
            help="Ablation: drop I_sem from the preference factor.",
        )
        parser.add_argument(
            "--unranking_no_contextual_correction", type=str2bool, default=False,
            help="Ablation: drop the PAPU contextual correction on D_inf \\ D_t.",
        )
        parser.add_argument(
            "--unranking_no_curvature", type=str2bool, default=False,
            help="Ablation: replace H_inf with the damping identity.",
        )
        parser.add_argument(
            "--unranking_dp", type=str2bool, default=False,
            help="Enable the optional privacy-oriented Gaussian-noise variant.",
        )
        parser.add_argument(
            "--unranking_dp_epsilon", type=float, default=1.0,
            help="Epsilon for the optional DP-style Gaussian perturbation.",
        )
        parser.add_argument(
            "--unranking_dp_delta", type=float, default=1e-5,
            help="Delta for the optional DP-style Gaussian perturbation.",
        )

        # Legacy RARU aliases retained for older scripts.
        parser.add_argument("--scope_cutoff_k", type=int, default=10,
                            help="Legacy / unused RARU scope threshold alias.")
        parser.add_argument("--damping_lambda", type=float, default=1e-2,
                            help="Legacy alias for --unranking_damping_lambda.")
        parser.add_argument("--cg_steps", type=int, default=30,
                            help="Legacy alias for --unranking_cg_steps.")
        parser.add_argument("--rarl_max_scope_size", type=int, default=5000,
                            help="Legacy / unused RARU scope cap alias.")
        parser.add_argument("--rarl_no_sconn", type=str2bool, default=False,
                            help="Legacy alias for --unranking_no_structural.")
        parser.add_argument("--rarl_no_sim", type=str2bool, default=False,
                            help="Legacy alias for --unranking_no_semantic.")
        parser.add_argument("--rarl_no_localized", type=str2bool, default=False,
                            help="Legacy alias for --unranking_no_scoping.")
        parser.add_argument("--rarl_no_salience_weight", type=str2bool, default=False,
                            help="Legacy alias for --unranking_no_preference_factor.")
        parser.add_argument("--rarl_no_margin", type=str2bool, default=False,
                            help="Legacy / unused RARU margin-preservation alias.")
        parser.add_argument("--rarl_no_curvature", type=str2bool, default=False,
                            help="Legacy alias for --unranking_no_curvature.")

        # UltraRE Parameters
        parser.add_argument("--ultrare_n_group", type=int, default=3,
                            help="Number of balanced clusters/sub-models for UltraRE.")
        parser.add_argument("--ultrare_sub_epochs", type=int, default=10,
                            help="Sub-model training epochs for UltraRE.")
        # Legacy aliases kept for backward compatibility.
        parser.add_argument("--ultrare_alpha", type=float, default=0.5,
                            help="(Legacy / unused) interpolation strength.")
        parser.add_argument("--ultrare_lr", type=float, default=1e-3,
                            help="(Legacy / unused) learning rate alias.")
        parser.add_argument("--ultrare_steps", type=int, default=50,
                            help="(Legacy / unused) fine-tune step count alias.")

        # RRL Parameters
        parser.add_argument("--rrl_epochs", type=int, default=5,
                            help="Number of reverse-BPR epochs for RRL.")
        parser.add_argument("--rrl_lr", type=float, default=1e-3,
                            help="Learning rate for RRL.")
        parser.add_argument("--rrl_batch_size", type=int, default=4096,
                            help="Batch size for RRL.")
        parser.add_argument("--rrl_fisher_eps", type=float, default=1e-4,
                            help="FIM regularizer for the inverse Fisher scaling in RRL.")
        parser.add_argument("--rrl_fisher_samples", type=int, default=4,
                            help="Mini-batches drawn from D_r to estimate the FIM diagonal.")

        # UnlearnRec Parameters
        parser.add_argument("--unlearnrec_epochs", type=int, default=5,
                            help="Fine-tuning epochs for UnlearnRec.")
        parser.add_argument("--unlearnrec_lr", type=float, default=1e-3,
                            help="Learning rate for UnlearnRec.")
        parser.add_argument("--unlearnrec_batch_size", type=int, default=4096,
                            help="Batch size for UnlearnRec.")
        parser.add_argument("--unlearnrec_unlearn_wei", type=float, default=1.0,
                            help="Weight of the unlearn loss (cal_neg_aug) in UnlearnRec.")
        parser.add_argument("--unlearnrec_align_wei", type=float, default=1.0,
                            help="Weight of the alignment loss in UnlearnRec.")
        # Legacy aliases.
        parser.add_argument("--unlearnrec_khop", type=int, default=1,
                            help="(Legacy / unused) k-hop subgraph radius.")
        parser.add_argument("--unlearnrec_lambda", type=float, default=0.5,
                            help="(Legacy / unused) forget-loss weight alias.")

        # GFEraser Parameters
        parser.add_argument("--gferaser_epochs", type=int, default=20,
                            help="Number of training epochs for the GFEraser PreFilter.")
        parser.add_argument("--gferaser_lr", type=float, default=1e-3,
                            help="Learning rate for GFEraser.")
        parser.add_argument("--gferaser_batch_size", type=int, default=1024,
                            help="Batch size for GFEraser.")
        parser.add_argument("--gferaser_cl_weight", type=float, default=0.4,
                            help="Weight of the contrastive (InfoNCE) loss in GFEraser.")
        parser.add_argument("--gferaser_pos_bpr_weight", type=float, default=1e-3,
                            help="Weight of the downstream attention-fused BPR loss.")
        parser.add_argument("--gferaser_temp", type=float, default=1.0,
                            help="Temperature for InfoNCE in GFEraser.")
        parser.add_argument("--gferaser_reg_weight", type=float, default=1e-4,
                            help="L2 weight on the original-graph embeddings in GFEraser.")

        return parser

