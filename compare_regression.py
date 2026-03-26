"""
Comparison experiment: Bar-distribution decoder vs MDN decoder.

Trains both models on the same TabICL prior settings and logs
loss / R² to TensorBoard so the runs can be overlaid.

Usage:
    uv run python compare_regression.py
    tensorboard --logdir runs/comparison
"""

import argparse
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.metrics import r2_score

from tfmplayground.callbacks import Callback
from tfmplayground.evaluation import get_openml_predictions, TOY_TASKS_REGRESSION
from tfmplayground.interface import NanoTabPFNRegressor, get_feature_preprocessor
from tfmplayground.mdn import MDNDecoder, MDNCriterion
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import TabICLPriorDataLoader
from tfmplayground.train import train
from tfmplayground.utils import (
    get_default_device,
    make_global_bucket_edges_from_prior,
    set_randomness_seed,
)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run_name",
    type=str,
    default="comparison",
    help="TensorBoard parent run dir under runs/",
)
parser.add_argument("--heads", type=int, default=6)
parser.add_argument("--embeddingsize", type=int, default=192)
parser.add_argument("--hiddensize", type=int, default=768)
parser.add_argument("--layers", type=int, default=6)
parser.add_argument(
    "--num_components", type=int, default=8, help="GMM components for MDN decoder"
)
parser.add_argument(
    "--n_buckets", type=int, default=100, help="Buckets for bar-distribution decoder"
)
parser.add_argument("--batchsize", type=int, default=1)
parser.add_argument("--accumulate", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--steps", type=int, default=100, help="steps per epoch")
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--num_datapoints_min", type=int, default=32)
parser.add_argument("--num_datapoints_max", type=int, default=128)
parser.add_argument("--min_features", type=int, default=1)
parser.add_argument("--max_features", type=int, default=50)
parser.add_argument("--max_num_classes", type=int, default=10)
parser.add_argument(
    "--prior_type",
    type=str,
    default="mix_scm",
    choices=["mlp_scm", "tree_scm", "mix_scm", "dummy"],
)
parser.add_argument(
    "--eval_every",
    type=int,
    default=10,
    help="run OpenML evaluation every N epochs (0 = disable)",
)
args = parser.parse_args()

SEED = 2402
device = get_default_device()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_prior():
    return TabICLPriorDataLoader(
        num_steps=args.steps,
        batch_size=args.batchsize,
        num_datapoints_min=args.num_datapoints_min,
        num_datapoints_max=args.num_datapoints_max,
        min_features=args.min_features,
        max_features=args.max_features,
        max_num_classes=args.max_num_classes,
        prior_type=args.prior_type,
        device=device,
    )


class MDNRegressor:
    """Minimal sklearn-compatible wrapper around a model with MDNDecoder."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def fit(self, X_train, y_train):
        self.preprocessor = get_feature_preprocessor(X_train)
        self.X_train = self.preprocessor.fit_transform(X_train)
        self.y_mean = float(np.mean(y_train))
        self.y_std = float(np.std(y_train, ddof=1)) + 1e-8
        self.y_train_n = (y_train - self.y_mean) / self.y_std

    def predict(self, X_test):
        X = np.concatenate((self.X_train, self.preprocessor.transform(X_test)))
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device).unsqueeze(0)
        y_t = torch.tensor(
            self.y_train_n, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            dist = self.model((X_t, y_t), single_eval_pos=len(self.X_train))
        return (dist.mean.squeeze(0) * self.y_std + self.y_mean).cpu().numpy()


# ---------------------------------------------------------------------------
# TensorBoard callback with optional R² eval
# ---------------------------------------------------------------------------


class ComparisonCallback(Callback):
    """
    Logs train NLL, epoch time, and (optionally) OpenML R² to TensorBoard.

    Args:
        log_dir:          SummaryWriter log directory
        label:            human-readable run name (printed to console)
        regressor_factory: callable(model, device) -> sklearn-compatible regressor
        eval_every:       run OpenML evaluation every N epochs (0 = never)
    """

    def __init__(
        self, log_dir: str, label: str, regressor_factory, eval_every: int = 10
    ):

        self.writer = SummaryWriter(log_dir=log_dir)
        self.label = label
        self.regressor_factory = regressor_factory
        self.eval_every = eval_every

    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        self.writer.add_scalar("Loss/train_nll", loss, epoch)
        self.writer.add_scalar("Time/epoch_seconds", epoch_time, epoch)

        if self.eval_every > 0 and epoch % self.eval_every == 0:
            model.eval()
            regressor = self.regressor_factory(model, device)
            predictions = get_openml_predictions(
                model=regressor, tasks=TOY_TASKS_REGRESSION
            )
            scores = [
                r2_score(y_true, y_pred)
                for _, (y_true, y_pred, _) in predictions.items()
            ]
            avg_r2 = sum(scores) / len(scores)
            self.writer.add_scalar("Eval/r2_mean", avg_r2, epoch)
            print(
                f"[{self.label}] epoch {epoch:5d} | time {epoch_time:5.2f}s"
                f" | nll {loss:.4f} | r² {avg_r2:.4f}",
                flush=True,
            )
        else:
            print(
                f"[{self.label}] epoch {epoch:5d} | time {epoch_time:5.2f}s | nll {loss:.4f}",
                flush=True,
            )

    def close(self):
        self.writer.close()


# ---------------------------------------------------------------------------
# Run 1: Bar-distribution decoder
# ---------------------------------------------------------------------------

print("=" * 60)
print("Run 1/2 — Bar-distribution decoder")
print("=" * 60)

set_randomness_seed(SEED)
prior_bar = make_prior()

bucket_edges = make_global_bucket_edges_from_prior(
    prior_bar, n_buckets=args.n_buckets, device=device
)
bar_dist = FullSupportBarDistribution(bucket_edges)

model_bar = NanoTabPFNModel(
    num_attention_heads=args.heads,
    embedding_size=args.embeddingsize,
    mlp_hidden_size=args.hiddensize,
    num_layers=args.layers,
    num_outputs=args.n_buckets,
)

bar_callback = ComparisonCallback(
    log_dir=f"runs/{args.run_name}/bar",
    label="bar",
    regressor_factory=lambda model, dev: NanoTabPFNRegressor(model, bar_dist, dev),
    eval_every=args.eval_every,
)

model_bar, _ = train(
    model=model_bar,
    prior=prior_bar,
    criterion=bar_dist,
    epochs=args.epochs,
    accumulate_gradients=args.accumulate,
    lr=args.lr,
    device=device,
    callbacks=[bar_callback],
    run_name=f"{args.run_name}_bar",
)

# ---------------------------------------------------------------------------
# Run 2: MDN decoder
# ---------------------------------------------------------------------------

print("=" * 60)
print("Run 2/2 — MDN decoder")
print("=" * 60)

set_randomness_seed(SEED)
prior_mdn = make_prior()

model_mdn = NanoTabPFNModel(
    num_attention_heads=args.heads,
    embedding_size=args.embeddingsize,
    mlp_hidden_size=args.hiddensize,
    num_layers=args.layers,
    num_outputs=1,  # placeholder — decoder replaced below
)
model_mdn.decoder = MDNDecoder(args.embeddingsize, args.hiddensize, args.num_components)

mdn_callback = ComparisonCallback(
    log_dir=f"runs/{args.run_name}/mdn",
    label="mdn",
    regressor_factory=lambda model, dev: MDNRegressor(model, dev),
    eval_every=args.eval_every,
)

model_mdn, _ = train(
    model=model_mdn,
    prior=prior_mdn,
    criterion=MDNCriterion(),
    epochs=args.epochs,
    accumulate_gradients=args.accumulate,
    lr=args.lr,
    device=device,
    callbacks=[mdn_callback],
    run_name=f"{args.run_name}_mdn",
)

# ---------------------------------------------------------------------------

print()
print("Training complete. View results with:")
print(f"  tensorboard --logdir runs/{args.run_name}")
