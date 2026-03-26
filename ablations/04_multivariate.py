"""
Ablation 4 — High-dimensional multivariate regression.

Task
----
    x ~ Uniform(-1, 1)^10   (10 features)
    y = sin(π·x₁)·cos(π·x₂)           signal pair 1
      + 0.5·x₃² − 0.4·x₄              signal pair 2
      + 0.3·x₅·x₆                     interaction term
      + 0.1·(x₇ + x₈ + x₉ + x₁₀)     weak linear features
      + σ(x)·ε                         heteroscedastic noise

    where σ(x) = 0.1 + 0.3·|sin(π·x₁)|   (noise tied to first feature)

Challenges at this level
------------------------
  - 10 input features, 6 of which carry genuine signal
  - Nonlinear interactions (x₁·x₂, x₃², x₅·x₆)
  - Heteroscedastic noise (σ depends on x₁)
  - MDN must generalise across a high-dimensional input space

We add a calibration check: for a range of nominal confidence levels α,
we compute the empirical coverage of the predicted [mean ± z_α·σ] interval.
A well-calibrated model's coverage curve lies on the diagonal.

Metrics logged
--------------
    - Test R², RMSE, mean NLL
    - 95% coverage
    - Calibration curve (nominal vs empirical)

Outputs
-------
    ablations/04_multivariate.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from scipy import stats

from tfmplayground.mdn import MDNCriterion, MDNDecoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_FEAT     = 10
N          = 6_000
N_EPOCHS   = 4_000
LR         = 3e-3
NUM_COMP   = 8
HIDDEN     = 128
SEED       = 3

torch.manual_seed(SEED); np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

X = torch.empty(N, N_FEAT).uniform_(-1.0, 1.0)

signal = (
    torch.sin(torch.pi * X[:, 0]) * torch.cos(torch.pi * X[:, 1])
    + 0.5 * X[:, 2] ** 2 - 0.4 * X[:, 3]
    + 0.3 * X[:, 4] * X[:, 5]
    + 0.1 * X[:, 6:].sum(dim=1)
)
noise_std = 0.1 + 0.3 * torch.abs(torch.sin(torch.pi * X[:, 0]))
y = signal + noise_std * torch.randn(N)

split = int(0.8 * N)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_model(num_components):
    enc = nn.Sequential(
        nn.Linear(N_FEAT, HIDDEN), nn.Tanh(),
        nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
        nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
    )
    dec = MDNDecoder(embedding_size=HIDDEN, mlp_hidden_size=HIDDEN, num_components=num_components)
    return enc, dec


def train(enc, dec, label):
    crit = MDNCriterion()
    opt  = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)
    history = []
    for epoch in range(1, N_EPOCHS + 1):
        dist = dec(enc(X_tr))
        loss = crit(dist, y_tr).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 200 == 0:
            history.append(loss.item())
    print(f"  [{label}] final NLL {history[-1]:.4f}")
    return history


def evaluate(enc, dec, label):
    enc.eval(); dec.eval()
    with torch.no_grad():
        d = dec(enc(X_te))
    pred = d.mean.numpy()
    std  = d.variance.sqrt().numpy()
    true = y_te.numpy()
    nll  = MDNCriterion()(d, y_te).mean().item()
    r2   = r2_score(true, pred)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    cov  = (np.abs(true - pred) <= 1.96 * std).mean()
    print(f"  [{label}]  R²={r2:.4f}  RMSE={rmse:.4f}  NLL={nll:.4f}  95%-cov={cov:.3f}")
    return pred, std, true, r2, rmse, nll, cov


def calibration_curve(pred, std, true, n_levels=20):
    """Empirical coverage at each nominal confidence level."""
    alphas = np.linspace(0.05, 0.99, n_levels)
    coverages = []
    for alpha in alphas:
        z = stats.norm.ppf((1 + alpha) / 2)
        coverages.append((np.abs(true - pred) <= z * std).mean())
    return alphas, np.array(coverages)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

print("Training MDN …")
enc_mdn,  dec_mdn  = make_model(NUM_COMP)
hist_mdn  = train(enc_mdn,  dec_mdn,  f"MDN k={NUM_COMP}")

print("Training Gaussian baseline …")
enc_base, dec_base = make_model(1)
hist_base = train(enc_base, dec_base, "Gaussian k=1")

print("\nTest metrics:")
pred_mdn,  std_mdn,  true_np, r2_mdn,  rmse_mdn,  nll_mdn,  cov_mdn  = evaluate(enc_mdn,  dec_mdn,  f"MDN k={NUM_COMP}")
pred_base, std_base, _,       r2_base, rmse_base, nll_base, cov_base = evaluate(enc_base, dec_base, "Gaussian k=1")

alphas_mdn,  covs_mdn  = calibration_curve(pred_mdn,  std_mdn,  true_np)
alphas_base, covs_base = calibration_curve(pred_base, std_base, true_np)

# ---------------------------------------------------------------------------
# Figure — 4 panels
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f"Ablation 4 — Multivariate regression ({N_FEAT}D input)", fontsize=13)

# Panel 1: parity plot (MDN)
ax = axes[0, 0]
lim = (true_np.min() - 0.2, true_np.max() + 0.2)
ax.scatter(true_np, pred_mdn,  s=4, alpha=0.3, color="steelblue", label=f"MDN  R²={r2_mdn:.3f}")
ax.scatter(true_np, pred_base, s=4, alpha=0.3, color="tomato", marker="x", label=f"Base R²={r2_base:.3f}")
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("True y"); ax.set_ylabel("Predicted y")
ax.set_title("Parity plot"); ax.legend(fontsize=9)

# Panel 2: residuals vs predicted std
ax = axes[0, 1]
residuals_mdn  = np.abs(true_np - pred_mdn)
residuals_base = np.abs(true_np - pred_base)
ax.scatter(std_mdn,  residuals_mdn,  s=3, alpha=0.2, color="steelblue", label=f"MDN k={NUM_COMP}")
ax.scatter(std_base, residuals_base, s=3, alpha=0.2, color="tomato",    label="Gaussian k=1")
max_val = max(std_mdn.max(), std_base.max(), residuals_mdn.max(), residuals_base.max())
ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="|residual| = σ")
ax.set_xlabel("Predicted σ"); ax.set_ylabel("|Residual|")
ax.set_title("Residuals vs predicted uncertainty"); ax.legend(fontsize=8)

# Panel 3: calibration curve
ax = axes[1, 0]
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="perfect calibration")
ax.plot(alphas_mdn,  covs_mdn,  color="steelblue", lw=2, marker="o", ms=4, label=f"MDN k={NUM_COMP}")
ax.plot(alphas_base, covs_base, color="tomato",    lw=2, marker="x", ms=4, label="Gaussian k=1")
ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage")
ax.set_title("Calibration (above diagonal = over-confident)"); ax.legend(fontsize=9)

# Panel 4: training NLL curves
ax = axes[1, 1]
epochs_logged = list(range(200, N_EPOCHS + 1, 200))
ax.plot(epochs_logged, hist_mdn,  color="steelblue", label=f"MDN k={NUM_COMP}")
ax.plot(epochs_logged, hist_base, color="tomato",    label="Gaussian k=1")
ax.set_xlabel("Epoch"); ax.set_ylabel("Train NLL")
ax.set_title("Training curves"); ax.legend(fontsize=9)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "04_multivariate.png")
plt.savefig(out, dpi=150); print(f"\nFigure saved → {out}")
try:
    plt.show()
except Exception:
    pass
