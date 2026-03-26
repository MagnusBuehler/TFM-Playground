"""
Ablation 3 — Heteroscedastic regression.

Task
----
    x ~ Uniform(-3, 3)   (1 feature)
    σ(x) = 0.05 + 0.5·x²          (noise grows away from origin)
    y = sin(π·x) + σ(x)·ε,   ε ~ N(0, 1)

The conditional mean is a simple sine wave, but the noise variance is
input-dependent. A Gaussian baseline (k=1) can learn one global σ that
approximates the average noise level, but it will be systematically
over-confident near x=0 and under-confident near the edges.

The MDN should learn a σₖ(x) that tracks the true σ(x) curve.
This is the key test: comparing the predicted uncertainty to ground truth.

Metrics logged
--------------
    - Test R², RMSE, mean NLL
    - Coverage: fraction of test points inside 95% predicted CI
    - σ(x) comparison: learned vs true

Outputs
-------
    ablations/03_heteroscedastic.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from tfmplayground.mdn import MDNCriterion, MDNDecoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N          = 4_000
N_EPOCHS   = 3_000
LR         = 3e-3
NUM_COMP   = 5
HIDDEN     = 64
SEED       = 2

torch.manual_seed(SEED); np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

x = torch.empty(N).uniform_(-3.0, 3.0)
true_sigma = 0.05 + 0.5 * x ** 2
y = torch.sin(torch.pi * x) + true_sigma * torch.randn(N)

split = int(0.8 * N)
x_tr, x_te = x[:split], x[split:]
y_tr, y_te = y[:split], y[split:]
X_tr = x_tr.unsqueeze(-1)
X_te = x_te.unsqueeze(-1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_model(num_components):
    enc = nn.Sequential(
        nn.Linear(1, HIDDEN), nn.Tanh(),
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
        if epoch % 100 == 0:
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
    # 95% coverage: fraction of true y within [mean - 1.96σ, mean + 1.96σ]
    within = np.abs(true - pred) <= 1.96 * std
    coverage = within.mean()
    print(f"  [{label}]  R²={r2:.4f}  RMSE={rmse:.4f}  NLL={nll:.4f}  95%-cov={coverage:.3f}")
    return pred, std, r2, rmse, nll, coverage

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
pred_mdn,  std_mdn,  r2_mdn,  rmse_mdn,  nll_mdn,  cov_mdn  = evaluate(enc_mdn,  dec_mdn,  f"MDN k={NUM_COMP}")
pred_base, std_base, r2_base, rmse_base, nll_base, cov_base = evaluate(enc_base, dec_base, "Gaussian k=1")

# ---------------------------------------------------------------------------
# Regression curve on a fine grid
# ---------------------------------------------------------------------------

x_vis     = torch.linspace(-3.0, 3.0, 400)
true_mean = torch.sin(torch.pi * x_vis).numpy()
true_sig  = (0.05 + 0.5 * x_vis ** 2).numpy()

enc_mdn.eval(); dec_mdn.eval()
enc_base.eval(); dec_base.eval()

with torch.no_grad():
    d_mdn  = dec_mdn (enc_mdn (x_vis.unsqueeze(-1)))
    d_base = dec_base(enc_base(x_vis.unsqueeze(-1)))

mu_mdn   = d_mdn.mean.numpy()
sig_mdn  = d_mdn.variance.sqrt().numpy()
mu_base  = d_base.mean.numpy()
sig_base = d_base.variance.sqrt().numpy()
x_vis_np = x_vis.numpy()

# ---------------------------------------------------------------------------
# Figure — 4 panels
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Ablation 3 — Heteroscedastic regression", fontsize=13)

# Panel 1: MDN regression curve + uncertainty band
ax = axes[0, 0]
ax.scatter(x_te.numpy(), y_te.numpy(), s=4, alpha=0.15, color="gray")
ax.plot(x_vis_np, true_mean, "k--", lw=1.5, label="true mean")
ax.plot(x_vis_np, mu_mdn, color="steelblue", lw=2, label=f"MDN k={NUM_COMP}")
ax.fill_between(x_vis_np, mu_mdn - 2*sig_mdn, mu_mdn + 2*sig_mdn,
                alpha=0.25, color="steelblue", label="±2σ")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(f"MDN — R²={r2_mdn:.3f}, 95%-cov={cov_mdn:.3f}"); ax.legend(fontsize=8)

# Panel 2: Gaussian baseline curve + uncertainty band
ax = axes[0, 1]
ax.scatter(x_te.numpy(), y_te.numpy(), s=4, alpha=0.15, color="gray")
ax.plot(x_vis_np, true_mean, "k--", lw=1.5, label="true mean")
ax.plot(x_vis_np, mu_base, color="tomato", lw=2, label="Gaussian k=1")
ax.fill_between(x_vis_np, mu_base - 2*sig_base, mu_base + 2*sig_base,
                alpha=0.25, color="tomato", label="±2σ")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(f"Gaussian — R²={r2_base:.3f}, 95%-cov={cov_base:.3f}"); ax.legend(fontsize=8)

# Panel 3: σ(x) comparison — learned vs true
ax = axes[1, 0]
ax.plot(x_vis_np, true_sig,  "k--", lw=2,   label="true σ(x)")
ax.plot(x_vis_np, sig_mdn,   color="steelblue", lw=2, label=f"MDN predicted σ(x)")
ax.plot(x_vis_np, sig_base,  color="tomato", lw=2, ls="--", label="Gaussian predicted σ(x)")
ax.set_xlabel("x"); ax.set_ylabel("σ")
ax.set_title("Predicted uncertainty vs ground truth"); ax.legend(fontsize=9)

# Panel 4: training NLL curves
ax = axes[1, 1]
epochs_logged = list(range(100, N_EPOCHS + 1, 100))
ax.plot(epochs_logged, hist_mdn,  color="steelblue", label=f"MDN k={NUM_COMP}")
ax.plot(epochs_logged, hist_base, color="tomato",    label="Gaussian k=1")
ax.set_xlabel("Epoch"); ax.set_ylabel("Train NLL")
ax.set_title("Training curves"); ax.legend(fontsize=9)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "03_heteroscedastic.png")
plt.savefig(out, dpi=150); print(f"\nFigure saved → {out}")
try:
    plt.show()
except Exception:
    pass
