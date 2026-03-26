"""
Ablation 5 — Real-world bimodal regression: Metro Interstate Traffic Volume.

Dataset
-------
    UCI Metro Interstate Traffic Volume (I-94 westbound, 2012-2018).
    CSV: ablations/Metro_Interstate_Traffic_Volume.csv

Task
----
    x = hour of day  ∈ {0, …, 23}   (single feature, normalised to [0, 1])
    y = traffic_volume               (standardised to zero mean / unit std)

    The day-of-week is deliberately withheld from the model, so at rush hours
    (6–9 am) the conditional p(y|x) is strongly bimodal:
        - weekday mode  ≈ high volume  (~6 000 vehicles/h at 7 am)
        - weekend mode  ≈ low  volume  (~1 600 vehicles/h at 7 am)

    A Gaussian (k=1) baseline is forced to predict the average of the two
    modes — which sits in the gap between them.  The MDN can place two
    separate components and achieve a much lower NLL.

Outputs
-------
    ablations/traffic/traffic_volume.png
"""

import os
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from tqdm import tqdm

from tfmplayground.mdn import MDNCriterion, MDNDecoder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    csv_path: str = os.path.join(
        os.path.dirname(__file__), "Metro_Interstate_Traffic_Volume.csv"
    )
    n_epochs: int = 50
    batch_size: int = 512
    lr: float = 3e-3
    num_components: int = 5
    hidden: int = 64
    seed: int = 1


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_data(cfg: Config):
    """
    Returns normalised tensors and the scalers needed to invert y for plotting.

    x: hour / 23  →  [0, 1]
    y: (traffic_volume − μ) / σ
    """
    import csv

    hours, volumes, is_weekend = [], [], []
    with open(cfg.csv_path) as f:
        for row in csv.DictReader(f):
            dt = datetime.strptime(row["date_time"], "%Y-%m-%d %H:%M:%S")
            hours.append(dt.hour)
            volumes.append(int(row["traffic_volume"]))
            is_weekend.append(dt.weekday() >= 5)

    x_raw = torch.tensor(hours, dtype=torch.float32) / 23.0  # O(N) normalise
    y_raw = torch.tensor(volumes, dtype=torch.float32)  # O(N)

    y_mean = y_raw.mean().item()
    y_std = y_raw.std().item()
    y_norm = (y_raw - y_mean) / y_std  # O(N) standardise

    is_weekend = torch.tensor(is_weekend)

    # 80/20 split (time-ordered to avoid leakage)
    n = len(x_raw)
    split = int(0.8 * n)
    X_tr = x_raw[:split].unsqueeze(-1)  # (N_tr, 1)
    y_tr = y_norm[:split]
    X_te = x_raw[split:].unsqueeze(-1)  # (N_te, 1)
    y_te = y_norm[split:]
    x_te_raw = x_raw[split:]
    is_weekend_te = is_weekend[split:]

    loader = DataLoader(  # O(1) construction; O(N) shuffle per epoch
        TensorDataset(X_tr, y_tr),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    return loader, X_te, y_te, x_te_raw, is_weekend_te, y_mean, y_std


def empirical_mode_curves(cfg: Config, y_mean: float, y_std: float):
    """
    Compute per-hour empirical means for weekdays and weekends separately.
    Returns arrays of shape (24,) in *normalised* y units.
    """
    import csv

    sums_wd = np.zeros(24)
    counts_wd = np.zeros(24)
    sums_we = np.zeros(24)
    counts_we = np.zeros(24)
    with open(cfg.csv_path) as f:
        for row in csv.DictReader(f):
            dt = datetime.strptime(row["date_time"], "%Y-%m-%d %H:%M:%S")
            h = dt.hour
            v = int(row["traffic_volume"])
            if dt.weekday() >= 5:
                sums_we[h] += v
                counts_we[h] += 1
            else:
                sums_wd[h] += v
                counts_wd[h] += 1

    mean_wd = (sums_wd / counts_wd - y_mean) / y_std
    mean_we = (sums_we / counts_we - y_mean) / y_std
    return mean_wd, mean_we


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def make_model(cfg: Config, num_components: int):
    # Parameter count: O(H + H² + H·K) — H² term dominates
    enc = nn.Sequential(
        nn.Linear(1, cfg.hidden),  # O(H) params
        nn.Tanh(),
        nn.Linear(cfg.hidden, cfg.hidden),  # O(H²) params
        nn.Tanh(),
    )
    dec = MDNDecoder(  # O(H·K) params
        embedding_size=cfg.hidden,
        mlp_hidden_size=cfg.hidden,
        num_components=num_components,
    )
    return enc, dec


def make_mlp(cfg: Config) -> nn.Module:
    # Same capacity as the encoder, but with a single scalar output head
    # Parameter count: O(H + H² + H) = O(H²)
    return nn.Sequential(
        nn.Linear(1, cfg.hidden),
        nn.Tanh(),
        nn.Linear(cfg.hidden, cfg.hidden),
        nn.Tanh(),
        nn.Linear(cfg.hidden, 1),  # point-prediction head
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(enc, dec, loader: DataLoader, cfg: Config, label: str):
    crit = MDNCriterion()
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=cfg.lr)
    history = []

    # Total training: O(E · N · H²)
    #   E epochs × N/B batches × O(B·H²) per batch (forward + backward)
    pbar = tqdm(range(1, cfg.n_epochs + 1), desc=label, unit="epoch")
    for epoch in pbar:
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:  # O(N/B) iterations per epoch
            dist = dec(enc(X_batch))  # forward:  O(B·H²)
            loss = crit(dist, y_batch).mean()  # NLL:      O(B·K)
            opt.zero_grad()
            loss.backward()  # backward: O(B·H²)
            opt.step()  # Adam:     O(P) = O(H²+H·K)
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0:
            avg = epoch_loss / n_batches
            history.append(avg)
            pbar.set_postfix(nll=f"{avg:.4f}")

    print(f"  [{label}] final NLL {history[-1]:.4f}")
    return history


def train_mlp(mlp: nn.Module, loader: DataLoader, cfg: Config, label: str):
    opt = torch.optim.Adam(mlp.parameters(), lr=cfg.lr)
    crit = nn.MSELoss()
    history = []

    pbar = tqdm(range(1, cfg.n_epochs + 1), desc=label, unit="epoch")
    for epoch in pbar:
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:
            pred = mlp(X_batch).squeeze(-1)  # (B,)
            loss = crit(pred, y_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0:
            avg = epoch_loss / n_batches
            history.append(avg)
            pbar.set_postfix(mse=f"{avg:.4f}")

    print(f"  [{label}] final MSE {history[-1]:.4f}")
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(enc, dec, X_te: torch.Tensor, y_te: torch.Tensor, label: str):
    enc.eval()
    dec.eval()
    with torch.no_grad():
        d = dec(enc(X_te))  # O(N_te · H²)
    pred = d.mean.numpy()
    true = y_te.numpy()
    nll = MDNCriterion()(d, y_te).mean().item()  # O(N_te · K)
    r2 = r2_score(true, pred)  # O(N_te)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))  # O(N_te)
    print(f"  [{label}]  R²={r2:.4f}  RMSE={rmse:.4f}  NLL={nll:.4f}")
    return pred, d.variance.sqrt().numpy(), r2, rmse, nll


def evaluate_mlp(mlp: nn.Module, X_te: torch.Tensor, y_te: torch.Tensor, label: str):
    mlp.eval()
    with torch.no_grad():
        pred = mlp(X_te).squeeze(-1).numpy()  # O(N_te · H²)
    true = y_te.numpy()
    r2 = r2_score(true, pred)  # O(N_te)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))  # O(N_te)
    print(f"  [{label}]  R²={r2:.4f}  RMSE={rmse:.4f}  NLL=n/a (point estimate)")
    return pred, r2, rmse


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def make_vis_grid(
    enc_mdn, dec_mdn, enc_base, dec_base, mlp, y_lo: float = -3.0, y_hi: float = 3.0
):
    """
    Returns a (H × Y) density grid for each model — the full p(y|x) surface.

    Grid sizes:
        H = 120 hour points  (x-axis of the heatmap)
        Y = 200 y points     (y-axis of the heatmap)

    Complexity: O(H · H²) forward pass + O(H · Y · K) density evaluation.
    """
    import math

    enc_mdn.eval()
    dec_mdn.eval()
    enc_base.eval()
    dec_base.eval()
    mlp.eval()

    H, Y = 120, 200
    x_vis = torch.linspace(0.0, 1.0, H).unsqueeze(-1)  # (H, 1)
    y_grid = torch.linspace(y_lo, y_hi, Y)  # (Y,)

    with torch.no_grad():
        d_mdn = dec_mdn(enc_mdn(x_vis))  # GMMDistribution, batch_shape (H,)
        d_base = dec_base(enc_base(x_vis))  # GMMDistribution, batch_shape (H,)
        mlp_pred = mlp(x_vis).squeeze(-1).numpy()  # (H,) point predictions

    def _density(dist, y_grid):
        """Evaluate GMM pdf over outer product (H hours) × (Y y-values) → (Y, H)."""
        pi, mu, sigma = dist.params  # each (H, K)
        y_e = y_grid[:, None, None]  # (Y, 1, 1)
        pi_e = pi[None]  # (1, H, K)
        mu_e = mu[None]  # (1, H, K)
        sigma_e = sigma[None]  # (1, H, K)
        comp = torch.exp(-0.5 * ((y_e - mu_e) / sigma_e) ** 2) / (
            sigma_e * math.sqrt(2 * math.pi)
        )
        return (pi_e * comp).sum(dim=-1).numpy()  # (Y, H)

    hours_np = x_vis.squeeze().numpy() * 23
    y_np = y_grid.numpy()

    return (
        hours_np,
        y_np,
        _density(d_mdn, y_grid),  # (Y, H)
        _density(d_base, y_grid),  # (Y, H)
        mlp_pred,  # (H,)
    )


# Consistent palette used across all panels
_C = {
    "mdn": "#4C9BE8",  # blue
    "gauss": "#E8634C",  # coral
    "mlp": "#4CE87A",  # green
    "weekday": "#FFD700",  # gold  — visible on dark heatmap background
    "weekend": "#FFB347",  # light orange
}


def _heatmap_panel(
    ax, fig, hours_np, y_np, density, mean_wd, mean_we, mlp_pred, title, vmax
):
    """
    p(y | hour) density heatmap with:
      - shared vmax across MDN / Gaussian panels for direct comparison
      - gold overlays for empirical weekday / weekend means
      - green dotted overlay for MLP point prediction
      - shaded rush-hour bands
    """
    mesh = ax.pcolormesh(
        hours_np, y_np, density, cmap="magma", shading="auto", vmin=0, vmax=vmax
    )
    fig.colorbar(mesh, ax=ax, label="p(y | hour)", pad=0.02)

    # Rush-hour shading (morning 6–9 am, evening 16–19)
    for lo, hi in [(6, 9), (16, 19)]:
        ax.axvspan(lo, hi, alpha=0.12, color="white", lw=0)

    hours_int = np.arange(24)
    ax.plot(hours_int, mean_wd, color=_C["weekday"], lw=2, label="weekday mean")
    ax.plot(
        hours_int, mean_we, color=_C["weekend"], lw=2, ls="--", label="weekend mean"
    )
    ax.plot(hours_np, mlp_pred, color=_C["mlp"], lw=1.5, ls=":", label="MLP prediction")

    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Traffic volume (standardised)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(
        fontsize=7,
        loc="upper left",
        framealpha=0.6,
        facecolor="#222222",
        labelcolor="white",
    )


def plot(
    cfg: Config,
    x_te_raw,
    y_te,
    pred_mdn,
    std_mdn,
    r2_mdn,
    pred_base,
    std_base,
    r2_base,
    pred_mlp,
    r2_mlp,
    hist_mdn,
    hist_base,
    hist_mlp,
    vis_grid,
    mean_wd,
    mean_we,
    out_path: str,
):
    hours_np, y_np, density_mdn, density_base, mlp_pred = vis_grid
    y_te_np = y_te.numpy()

    # Shared density scale so heatmaps are directly comparable
    vmax = max(density_mdn.max(), density_base.max())

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        "Ablation 5 — Metro traffic volume  ·  bimodal p(y | hour): weekday vs weekend",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    ax_mdn = fig.add_subplot(gs[0, 0])
    ax_gauss = fig.add_subplot(gs[0, 1], sharey=ax_mdn)
    ax_par = fig.add_subplot(gs[1, 0])
    ax_loss = fig.add_subplot(gs[1, 1])

    # ── Row 0: density heatmaps ───────────────────────────────────────────
    _heatmap_panel(
        ax_mdn,
        fig,
        hours_np,
        y_np,
        density_mdn,
        mean_wd,
        mean_we,
        mlp_pred,
        title=f"MDN  k={cfg.num_components}  —  p(y | hour)",
        vmax=vmax,
    )
    _heatmap_panel(
        ax_gauss,
        fig,
        hours_np,
        y_np,
        density_base,
        mean_wd,
        mean_we,
        mlp_pred,
        title="Gaussian  k=1  —  p(y | hour)",
        vmax=vmax,
    )
    # hide duplicate y-axis label on the shared right panel
    ax_gauss.set_ylabel("")

    # ── Row 1, left: parity plot ──────────────────────────────────────────
    ax = ax_par
    lim = (y_te_np.min() - 0.3, y_te_np.max() + 0.3)

    ax.scatter(y_te_np, pred_base, s=2, alpha=0.12, color=_C["gauss"], rasterized=True)
    ax.scatter(y_te_np, pred_mlp, s=2, alpha=0.12, color=_C["mlp"], rasterized=True)
    ax.scatter(y_te_np, pred_mdn, s=2, alpha=0.15, color=_C["mdn"], rasterized=True)

    # Invisible large markers just for a readable legend
    for color, label in [
        (_C["mdn"], f"MDN  k={cfg.num_components}   R²={r2_mdn:.3f}"),
        (_C["gauss"], f"Gaussian k=1   R²={r2_base:.3f}"),
        (_C["mlp"], f"MLP (MSE)      R²={r2_mlp:.3f}"),
    ]:
        ax.scatter([], [], s=40, color=color, label=label)

    ax.plot(lim, lim, color="0.3", lw=1, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("True y (standardised)")
    ax.set_ylabel("Predicted y (standardised)")
    ax.set_title("Parity plot", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.set_aspect("equal", adjustable="box")

    # ── Row 1, right: NLL training curves (MDN vs Gaussian only) ─────────
    ax = ax_loss
    epochs_logged = list(range(10, cfg.n_epochs + 1, 10))
    ax.plot(
        epochs_logged,
        hist_mdn,
        color=_C["mdn"],
        lw=2,
        label=f"MDN k={cfg.num_components}",
    )
    ax.plot(epochs_logged, hist_base, color=_C["gauss"], lw=2, label="Gaussian k=1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train NLL")
    ax.set_title("NLL training curves", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # MLP uses MSE (different scale) — annotate final value instead of dual axis
    ax.text(
        0.97,
        0.97,
        f"MLP final MSE = {hist_mlp[-1]:.4f}\n(trained with MSE, not NLL)",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#eef", alpha=0.8),
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {out_path}")
    try:
        plt.show()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    loader, X_te, y_te, x_te_raw, is_weekend_te, y_mean, y_std = load_data(cfg)
    mean_wd, mean_we = empirical_mode_curves(cfg, y_mean, y_std)

    print("Training MDN …")
    enc_mdn, dec_mdn = make_model(cfg, cfg.num_components)
    hist_mdn = train(enc_mdn, dec_mdn, loader, cfg, f"MDN k={cfg.num_components}")

    print("Training Gaussian baseline …")
    enc_base, dec_base = make_model(cfg, 1)
    hist_base = train(enc_base, dec_base, loader, cfg, "Gaussian k=1")

    print("Training MLP …")
    mlp = make_mlp(cfg)
    hist_mlp = train_mlp(mlp, loader, cfg, "MLP")

    print("\nTest metrics:")
    pred_mdn, std_mdn, r2_mdn, *_ = evaluate(
        enc_mdn, dec_mdn, X_te, y_te, f"MDN k={cfg.num_components}"
    )
    pred_base, std_base, r2_base, *_ = evaluate(
        enc_base, dec_base, X_te, y_te, "Gaussian k=1"
    )
    pred_mlp, r2_mlp, *_ = evaluate_mlp(mlp, X_te, y_te, "MLP")

    vis_grid = make_vis_grid(enc_mdn, dec_mdn, enc_base, dec_base, mlp)

    out_dir = os.path.join(os.path.dirname(__file__), "traffic")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "traffic_volume.png")

    plot(
        cfg,
        x_te_raw,
        y_te,
        pred_mdn,
        std_mdn,
        r2_mdn,
        pred_base,
        std_base,
        r2_base,
        pred_mlp,
        r2_mlp,
        hist_mdn,
        hist_base,
        hist_mlp,
        vis_grid,
        mean_wd,
        mean_we,
        out_path,
    )


if __name__ == "__main__":
    main()
