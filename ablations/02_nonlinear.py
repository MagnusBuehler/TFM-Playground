"""
Ablation 2 — Nonlinear regression.

Task
----
    x ~ Uniform(-3, 3)   (1 feature)
    y = 0.5·x³ − 2·sin(π·x) + 0.4·ε,   ε ~ N(0, 1)

The relationship is a smooth but strongly nonlinear curve. Noise is still
homoscedastic so both models should capture the mean well. The MDN's
advantage here is purely in fitting a complex regression function —
extra components act as a richer basis for the conditional mean.

Because input is 1D we can visualise the full regression curve with
a ±2σ uncertainty band, making it easy to see whether the model
tracks the true function.

Metrics logged
--------------
    - Test R², RMSE, mean NLL
    - Training NLL curves (MDN vs Gaussian baseline)

Outputs
-------
    ablations/nonlinear/nonlinear_{mode}.png
"""

import os
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

from tfmplayground.mdn import MDNCriterion, MDNDecoder


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    n: int = 3_000
    n_epochs: int = 2000
    batch_size: int = 256
    lr: float = 3e-3
    num_components: int = 5
    hidden: int = 64
    seed: int = 1
    mode: Literal["easy", "moderate", "hard", "bimodal", "trimodal", "10-modal"] = (
        "easy"
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


_MODAL_MODES = {"bimodal": 2, "trimodal": 3, "10-modal": 10}
_MODAL_SEP = {"bimodal": 6.0, "trimodal": 5.0, "10-modal": 2.5}


def _modal_offsets(mode: str) -> torch.Tensor:
    """Evenly spaced offsets centred on zero, one per mixture component."""
    k = _MODAL_MODES[mode]
    sep = _MODAL_SEP[mode]
    return torch.linspace(-(k - 1) * sep / 2, (k - 1) * sep / 2, k)


def _ground_truth(x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    easy     — smooth cubic + low-freq sine, homoscedastic noise σ=0.4
    moderate — adds a medium-freq oscillation (3πx) and doubles the noise
    hard     — stacks a high-freq term (7πx) on top and uses heteroscedastic
               noise that grows with |x|, making the tails very noisy
    bimodal  — each sample drawn from one of 2 sine-shifted strands (sep=6)
    trimodal — each sample drawn from one of 3 sine-shifted strands (sep=5)
    10-modal — each sample drawn from one of 10 sine-shifted strands (sep=2.5)
    """
    base = 0.5 * x**3 - 2.0 * torch.sin(torch.pi * x)
    if mode == "easy":
        return base + 0.4 * torch.randn_like(x)
    if mode == "moderate":
        return base + 1.5 * torch.sin(3 * torch.pi * x) + 1.2 * torch.randn_like(x)
    if mode == "hard":
        noise_std = 1.0 + 1.5 * x.abs()  # heteroscedastic: σ ∈ [1, 5.5] over [-3,3]
        return (
            base
            + 1.5 * torch.sin(3 * torch.pi * x)
            + torch.sin(7 * torch.pi * x)
            + noise_std * torch.randn_like(x)
        )
    # multimodal: assign each point to a random component, add its offset
    offsets = _modal_offsets(mode)
    k = len(offsets)
    idx = torch.randint(0, k, (len(x),))
    return torch.sin(torch.pi * x) + offsets[idx] + 0.3 * torch.randn_like(x)


def make_data(cfg: Config):
    x = torch.empty(cfg.n).uniform_(-3.0, 3.0)  # O(N)
    y = _ground_truth(x, cfg.mode)  # O(N)

    split = int(0.8 * cfg.n)
    x_tr, x_te = x[:split], x[split:]
    y_tr, y_te = y[:split], y[split:]
    X_tr = x_tr.unsqueeze(-1)  # (N_tr, 1)
    X_te = x_te.unsqueeze(-1)  # (N_te, 1)

    loader = DataLoader(  # O(1) — lazy; shuffled index built per epoch in O(N)
        TensorDataset(X_tr, y_tr),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    return loader, X_te, y_te, x_te


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def make_model(cfg: Config, num_components: int):
    # Parameter count: O(H + H² + H·K) — the H² term (hidden→hidden) dominates
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
            dist = dec(enc(X_batch))  # forward: O(B·H²) — hidden→hidden dominates
            loss = crit(dist, y_batch).mean()  # NLL: O(B·K) per batch
            opt.zero_grad()
            loss.backward()  # backward: O(B·H²) — mirrors forward
            opt.step()  # Adam update: O(P) = O(H² + H·K)
            epoch_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0:
            avg = epoch_loss / n_batches
            history.append(avg)
            pbar.set_postfix(nll=f"{avg:.4f}")

    print(f"  [{label}] final NLL {history[-1]:.4f}")
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(enc, dec, X_te: torch.Tensor, y_te: torch.Tensor, label: str):
    enc.eval()
    dec.eval()
    with torch.no_grad():
        d = dec(enc(X_te))  # O(N_te · H²) — single forward pass over test set
    pred = d.mean.numpy()
    true = y_te.numpy()
    nll = MDNCriterion()(d, y_te).mean().item()  # O(N_te · K)
    r2 = r2_score(true, pred)  # O(N_te)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))  # O(N_te)
    print(f"  [{label}]  R²={r2:.4f}  RMSE={rmse:.4f}  NLL={nll:.4f}")
    return pred, d.variance.sqrt().numpy(), r2, rmse, nll


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def make_vis_grid(enc_mdn, dec_mdn, enc_base, dec_base, cfg: Config):
    enc_mdn.eval()
    dec_mdn.eval()
    enc_base.eval()
    dec_base.eval()

    x_vis = torch.linspace(-3.0, 3.0, 400)

    # Noiseless ground-truth mean curves — one array per mode component.
    base = 0.5 * x_vis**3 - 2.0 * torch.sin(torch.pi * x_vis)
    if cfg.mode == "easy":
        true_curves = [base.numpy()]  # O(400) = O(1)
    elif cfg.mode == "moderate":
        true_curves = [(base + 1.5 * torch.sin(3 * torch.pi * x_vis)).numpy()]
    elif cfg.mode == "hard":
        true_curves = [
            (
                base
                + 1.5 * torch.sin(3 * torch.pi * x_vis)
                + torch.sin(7 * torch.pi * x_vis)
            ).numpy()
        ]
    else:  # multimodal — one noiseless curve per component
        base_modal = torch.sin(torch.pi * x_vis)
        offsets = _modal_offsets(cfg.mode)
        true_curves = [(base_modal + off).numpy() for off in offsets]

    with torch.no_grad():
        d_mdn = dec_mdn(
            enc_mdn(x_vis.unsqueeze(-1))
        )  # O(400 · H²) = O(H²) — fixed grid
        d_base = dec_base(enc_base(x_vis.unsqueeze(-1)))  # O(400 · H²) = O(H²)

    return (
        x_vis.numpy(),
        true_curves,
        d_mdn.mean.numpy(),
        d_mdn.variance.sqrt().numpy(),
        d_base.mean.numpy(),
        d_base.variance.sqrt().numpy(),
    )


def plot(
    cfg: Config,
    x_te,
    y_te,
    pred_mdn,
    std_mdn,
    r2_mdn,
    pred_base,
    std_base,
    r2_base,
    hist_mdn,
    hist_base,
    vis_grid,
    out_path: str,
):
    x_vis_np, true_curves, mu_mdn, s_mdn, mu_base, s_base = vis_grid
    y_te_np = y_te.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Ablation 2 — Nonlinear regression ({cfg.mode})", fontsize=13)

    # Panel 1: regression curve
    ax = axes[0]
    ax.scatter(x_te.numpy(), y_te_np, s=4, alpha=0.2, color="gray", label="test data")
    for i, curve in enumerate(true_curves):
        ax.plot(
            x_vis_np,
            curve,
            "k--",
            lw=1.0,
            alpha=0.6,
            label="true modes" if i == 0 else None,
        )
    ax.plot(
        x_vis_np, mu_mdn, color="steelblue", lw=2, label=f"MDN k={cfg.num_components}"
    )
    ax.fill_between(
        x_vis_np, mu_mdn - 2 * s_mdn, mu_mdn + 2 * s_mdn, alpha=0.2, color="steelblue"
    )
    ax.plot(x_vis_np, mu_base, color="tomato", lw=2, ls="--", label="Gaussian k=1")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Regression curve ± 2σ")
    ax.legend(fontsize=8)

    # Panel 2: parity plot
    ax = axes[1]
    lim = (y_te_np.min() - 0.5, y_te_np.max() + 0.5)
    ax.scatter(
        y_te_np,
        pred_mdn,
        s=5,
        alpha=0.4,
        color="steelblue",
        label=f"MDN  R²={r2_mdn:.3f}",
    )
    ax.scatter(
        y_te_np,
        pred_base,
        s=5,
        alpha=0.4,
        color="tomato",
        marker="x",
        label=f"Base R²={r2_base:.3f}",
    )
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("True y")
    ax.set_ylabel("Predicted y")
    ax.set_title("Parity plot")
    ax.legend(fontsize=9)

    # Panel 3: training curves
    ax = axes[2]
    epochs_logged = list(range(10, cfg.n_epochs + 1, 10))
    ax.plot(
        epochs_logged, hist_mdn, color="steelblue", label=f"MDN k={cfg.num_components}"
    )
    ax.plot(epochs_logged, hist_base, color="tomato", label="Gaussian k=1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train NLL")
    ax.set_title("Training curve")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved → {out_path}")
    try:
        plt.show()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = Config(mode="bimodal")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    loader, X_te, y_te, x_te = make_data(cfg)

    print("Training MDN …")
    enc_mdn, dec_mdn = make_model(cfg, cfg.num_components)
    hist_mdn = train(enc_mdn, dec_mdn, loader, cfg, f"MDN k={cfg.num_components}")

    print("Training Gaussian baseline …")
    enc_base, dec_base = make_model(cfg, 1)
    hist_base = train(enc_base, dec_base, loader, cfg, "Gaussian k=1")

    print("\nTest metrics:")
    pred_mdn, std_mdn, r2_mdn, rmse_mdn, nll_mdn = evaluate(
        enc_mdn, dec_mdn, X_te, y_te, f"MDN k={cfg.num_components}"
    )
    pred_base, std_base, r2_base, rmse_base, nll_base = evaluate(
        enc_base, dec_base, X_te, y_te, "Gaussian k=1"
    )

    vis_grid = make_vis_grid(enc_mdn, dec_mdn, enc_base, dec_base, cfg)

    out_dir = os.path.join(os.path.dirname(__file__), "nonlinear")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"nonlinear_{cfg.mode}_{cfg.n_epochs}.png")
    plot(
        cfg,
        x_te,
        y_te,
        pred_mdn,
        std_mdn,
        r2_mdn,
        pred_base,
        std_base,
        r2_base,
        hist_mdn,
        hist_base,
        vis_grid,
        out_path,
    )


if __name__ == "__main__":
    main()
