"""
Toy experiment: fitting a bimodal conditional distribution with MDNDecoder.

True generative process
-----------------------
    x ~ Uniform(-1, 1)
    branch ~ Bernoulli(0.5)
    y | x, branch=0  ~  N( 2x,  0.2²)   ← upper branch
    y | x, branch=1  ~  N(-2x,  0.2²)   ← lower branch

A single-Gaussian decoder collapses both modes and predicts y ≈ 0 for all x.
The MDN decoder learns to assign separate components to each branch.

Outputs
-------
    ablations/fit_multimodal.png   — 4-panel figure
    (also shown on screen if a display is available)

Run
---
    uv run python ablations/fit_multimodal.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from tfmplayground.mdn import GMMDistribution, MDNCriterion, MDNDecoder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_TRAIN = 4_000
N_EPOCHS = 2_000
LR = 3e-3
NUM_COMPONENTS = 5
HIDDEN = 64
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Synthetic data: bimodal conditional distribution
# ---------------------------------------------------------------------------

x_train = torch.empty(N_TRAIN).uniform_(-1.0, 1.0)
branch = torch.bernoulli(0.5 * torch.ones(N_TRAIN)).bool()
y_train = torch.where(
    branch,
     2.0 * x_train + 0.2 * torch.randn(N_TRAIN),
    -2.0 * x_train + 0.2 * torch.randn(N_TRAIN),
)
# shape: (N,) for both x and y

x_input = x_train.unsqueeze(-1)  # (N, 1) — feature dimension expected by Linear

# ---------------------------------------------------------------------------
# Model: tiny MLP encoder  +  MDNDecoder from the main pipeline
# ---------------------------------------------------------------------------

encoder = nn.Sequential(
    nn.Linear(1, HIDDEN),
    nn.Tanh(),
    nn.Linear(HIDDEN, HIDDEN),
    nn.Tanh(),
)
decoder = MDNDecoder(embedding_size=HIDDEN, mlp_hidden_size=HIDDEN, num_components=NUM_COMPONENTS)
criterion = MDNCriterion()

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=LR
)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print(f"Training MDN ({NUM_COMPONENTS} components) for {N_EPOCHS} epochs …")
for epoch in range(1, N_EPOCHS + 1):
    hidden = encoder(x_input)          # (N, HIDDEN)
    dist = decoder(hidden)             # GMMDistribution, batch shape (N,)
    loss = criterion(dist, y_train).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"  epoch {epoch:5d} | NLL {loss.item():.4f}")

print("Done.")

# ---------------------------------------------------------------------------
# Evaluation on a fine x grid for visualisation
# ---------------------------------------------------------------------------

encoder.eval()
decoder.eval()

x_vis = torch.linspace(-1.0, 1.0, 200)
with torch.no_grad():
    h_vis = encoder(x_vis.unsqueeze(-1))   # (200, HIDDEN)
    d_vis: GMMDistribution = decoder(h_vis)

pi_vis   = d_vis.pi.numpy()    # (200, K)
mu_vis   = d_vis.mu.numpy()    # (200, K)
sigma_vis = d_vis.sigma.numpy()  # (200, K)

# Density heatmap: p(y | x) over a (y_grid × x_grid) mesh
y_vis = torch.linspace(-3.5, 3.5, 300)
# broadcast: (300, 1) evaluated under GMM with batch shape (200,)
density = np.zeros((300, 200))
for xi in range(200):
    p = pi_vis[xi]       # (K,)
    m = mu_vis[xi]       # (K,)
    s = sigma_vis[xi]    # (K,)
    # sum_k π_k * N(y | μ_k, σ_k)
    y_np = y_vis.numpy()[:, None]  # (300, 1)
    component = np.exp(-0.5 * ((y_np - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    density[:, xi] = (p * component).sum(axis=1)

# ---------------------------------------------------------------------------
# 4-panel figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("MDN fit to a bimodal conditional distribution", fontsize=14)

x_np = x_train.numpy()
y_np_train = y_train.numpy()
x_vis_np = x_vis.numpy()

# ── Panel 1: training data ────────────────────────────────────────────────
ax = axes[0, 0]
ax.scatter(x_np[~branch.numpy()], y_np_train[~branch.numpy()],
           s=2, alpha=0.3, color="steelblue", label="branch 0 (+2x)")
ax.scatter(x_np[branch.numpy()],  y_np_train[branch.numpy()],
           s=2, alpha=0.3, color="tomato",   label="branch 1 (−2x)")
ax.set_title("Training data")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(markerscale=4, fontsize=8)

# ── Panel 2: learned density heatmap  p(y | x) ────────────────────────────
ax = axes[0, 1]
im = ax.imshow(
    density,
    origin="lower",
    aspect="auto",
    extent=[-1, 1, -3.5, 3.5],
    cmap="viridis",
    interpolation="bilinear",
)
fig.colorbar(im, ax=ax, label="p(y | x)")
ax.set_title("Learned density  p(y | x)")
ax.set_xlabel("x")
ax.set_ylabel("y")

# ── Panel 3: component means μ_k(x) ──────────────────────────────────────
ax = axes[1, 0]
cmap = plt.cm.tab10
for k in range(NUM_COMPONENTS):
    ax.plot(x_vis_np, mu_vis[:, k], color=cmap(k), label=f"μ_{k}")
ax.set_title("Component means  μₖ(x)")
ax.set_xlabel("x")
ax.set_ylabel("μₖ")
ax.legend(fontsize=8)

# ── Panel 4: mixing weights π_k(x) ───────────────────────────────────────
ax = axes[1, 1]
for k in range(NUM_COMPONENTS):
    ax.plot(x_vis_np, pi_vis[:, k], color=cmap(k), label=f"π_{k}")
ax.set_ylim(0, 1)
ax.set_title("Mixing weights  πₖ(x)")
ax.set_xlabel("x")
ax.set_ylabel("πₖ")
ax.legend(fontsize=8)

plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "fit_multimodal.png")
plt.savefig(out_path, dpi=150)
print(f"Figure saved → {out_path}")

try:
    plt.show()
except Exception:
    pass
