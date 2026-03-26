"""
Ablation 1 — Linear regression (baseline difficulty).

Task
----
    x ~ Uniform(-3, 3)²   (2 features)
    y = 2·x₁ − x₂ + 0.5·ε,   ε ~ N(0, 1)

The ground-truth relationship is perfectly linear with homoscedastic noise.
A single Gaussian decoder (k=1) should solve this; an MDN with k>1 should
match it without being penalised — all excess components should collapse to
the same mean with near-zero weight.

This is the sanity-check level: both models should achieve near-identical R².

Metrics logged
--------------
    - Test R², RMSE, mean NLL
    - Training NLL curve (MDN vs baseline)

Outputs
-------
    ablations/01_linear.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

from tfmplayground.mdn import GMMDistribution, MDNCriterion, MDNDecoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N          = 3_000
N_EPOCHS   = 2_000
LR         = 3e-3
NUM_COMP   = 5      # MDN
HIDDEN     = 64
SEED       = 0

torch.manual_seed(SEED); np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

X = torch.empty(N, 2).uniform_(-3.0, 3.0)
y = 2.0 * X[:, 0] - X[:, 1] + 0.5 * torch.randn(N)

split = int(0.8 * N)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

# ---------------------------------------------------------------------------
# Build a model given num_components
# ---------------------------------------------------------------------------

def make_model(num_components):
    enc = nn.Sequential(
        nn.Linear(2, HIDDEN), nn.Tanh(),
        nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
    )
    dec = MDNDecoder(embedding_size=HIDDEN, mlp_hidden_size=HIDDEN, num_components=num_components)
    return enc, dec

# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Train MDN and Gaussian baseline
# ---------------------------------------------------------------------------

print("Training MDN …")
enc_mdn,  dec_mdn  = make_model(NUM_COMP)
hist_mdn  = train(enc_mdn,  dec_mdn,  f"MDN k={NUM_COMP}")

print("Training Gaussian baseline …")
enc_base, dec_base = make_model(1)
hist_base = train(enc_base, dec_base, "Gaussian k=1")

# ---------------------------------------------------------------------------
# Evaluation on test set
# ---------------------------------------------------------------------------

def evaluate(enc, dec, label):
    enc.eval(); dec.eval()
    with torch.no_grad():
        d = dec(enc(X_te))
    pred = d.mean.numpy()
    true = y_te.numpy()
    nll  = MDNCriterion()(d, y_te).mean().item()
    r2   = r2_score(true, pred)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    print(f"  [{label}]  R²={r2:.4f}  RMSE={rmse:.4f}  NLL={nll:.4f}")
    return pred, d.variance.sqrt().numpy(), r2, rmse, nll

print("\nTest metrics:")
pred_mdn,  std_mdn,  r2_mdn,  rmse_mdn,  nll_mdn  = evaluate(enc_mdn,  dec_mdn,  f"MDN k={NUM_COMP}")
pred_base, std_base, r2_base, rmse_base, nll_base = evaluate(enc_base, dec_base, "Gaussian k=1")

# ---------------------------------------------------------------------------
# Figure — 3 panels
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Ablation 1 — Linear regression", fontsize=13)

true_np = y_te.numpy()

# Panel 1: parity plot
ax = axes[0]
lim = (true_np.min() - 0.3, true_np.max() + 0.3)
ax.scatter(true_np, pred_mdn,  s=6, alpha=0.4, label=f"MDN  R²={r2_mdn:.3f}",  color="steelblue")
ax.scatter(true_np, pred_base, s=6, alpha=0.4, label=f"Base R²={r2_base:.3f}", color="tomato", marker="x")
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("True y"); ax.set_ylabel("Predicted y")
ax.set_title("Parity plot"); ax.legend(fontsize=9)

# Panel 2: predicted uncertainty histogram
ax = axes[1]
ax.hist(std_mdn,  bins=40, alpha=0.6, color="steelblue", label=f"MDN k={NUM_COMP}")
ax.hist(std_base, bins=40, alpha=0.6, color="tomato",    label="Gaussian k=1")
ax.set_xlabel("Predicted std"); ax.set_ylabel("Count")
ax.set_title("Uncertainty distribution"); ax.legend(fontsize=9)

# Panel 3: training NLL curves
ax = axes[2]
epochs_logged = list(range(100, N_EPOCHS + 1, 100))
ax.plot(epochs_logged, hist_mdn,  color="steelblue", label=f"MDN k={NUM_COMP}")
ax.plot(epochs_logged, hist_base, color="tomato",    label="Gaussian k=1")
ax.set_xlabel("Epoch"); ax.set_ylabel("Train NLL")
ax.set_title("Training curve"); ax.legend(fontsize=9)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "01_linear.png")
plt.savefig(out, dpi=150); print(f"\nFigure saved → {out}")
try:
    plt.show()
except Exception:
    pass
