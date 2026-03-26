"""
Setting 3 — Mode splitting: the number of active modes changes with x.

True generative process
-----------------------
    x ~ Uniform(-1.5, 1.5), split into three regimes:

    x < -0.5   (regime 1, unimodal)
        y ~ N(0, 0.2²)

    -0.5 ≤ x < 0.5   (regime 2, bimodal)
        y ~ 0.5·N(-1, 0.15²)  +  0.5·N(+1, 0.15²)

    x ≥ 0.5   (regime 3, trimodal)
        y ~ N(-1.5, 0.15²)/3  +  N(0, 0.15²)/3  +  N(+1.5, 0.15²)/3

The boundaries at x = ±0.5 are sharp — the distribution literally gains new modes
as x crosses them. A fixed-K Gaussian mixture can represent this if it learns to
zero out the excess weights in regime 1 and split weight evenly in regime 3.

What to look for
----------------
- Density heatmap: single stripe → two stripes → three stripes as x increases
- Mixing weights π_k(x): components switch on/off at the regime boundaries
- The model needs ≥ 3 components to represent all regimes simultaneously

Run
---
    uv run python ablations/03_mode_splitting.py
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

N_TRAIN       = 6_000
N_EPOCHS      = 3_000
LR            = 3e-3
NUM_COMPONENTS = 6
HIDDEN        = 128
SEED          = 1

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

x_train = torch.empty(N_TRAIN).uniform_(-1.5, 1.5)

regime = torch.where(x_train < -0.5,
                     torch.zeros_like(x_train),
         torch.where(x_train <  0.5,
                     torch.ones_like(x_train),
                     2 * torch.ones_like(x_train))).long()

noise = torch.randn(N_TRAIN)
u     = torch.rand(N_TRAIN)

# Regime 1: unimodal at 0
r1 = torch.zeros(N_TRAIN) + 0.2 * noise

# Regime 2: bimodal at ±1
r2 = torch.where(u < 0.5, -1.0 + 0.15 * noise, 1.0 + 0.15 * noise)

# Regime 3: trimodal at -1.5, 0, +1.5
r3 = torch.where(u < 1/3, -1.5 + 0.15 * noise,
     torch.where(u < 2/3,  0.0 + 0.15 * noise,
                            1.5 + 0.15 * noise))

y_train = torch.where(regime == 0, r1, torch.where(regime == 1, r2, r3))

x_input = x_train.unsqueeze(-1)  # (N, 1)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

encoder = nn.Sequential(
    nn.Linear(1, HIDDEN), nn.Tanh(),
    nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
)
decoder   = MDNDecoder(embedding_size=HIDDEN, mlp_hidden_size=HIDDEN, num_components=NUM_COMPONENTS)
criterion = MDNCriterion()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

print(f"Training MDN ({NUM_COMPONENTS} components) for {N_EPOCHS} epochs …")
for epoch in range(1, N_EPOCHS + 1):
    dist = decoder(encoder(x_input))
    loss = criterion(dist, y_train).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"  epoch {epoch:5d} | NLL {loss.item():.4f}")
print("Done.")

# ---------------------------------------------------------------------------
# Evaluation grid
# ---------------------------------------------------------------------------

encoder.eval(); decoder.eval()

x_vis = torch.linspace(-1.5, 1.5, 300)
with torch.no_grad():
    d_vis: GMMDistribution = decoder(encoder(x_vis.unsqueeze(-1)))

pi_vis    = d_vis.pi.numpy()
mu_vis    = d_vis.mu.numpy()
sigma_vis = d_vis.sigma.numpy()
x_vis_np  = x_vis.numpy()

y_vis   = torch.linspace(-2.5, 2.5, 400)
density = np.zeros((400, 300))
for xi in range(300):
    p  = pi_vis[xi];  m = mu_vis[xi];  s = sigma_vis[xi]
    yy = y_vis.numpy()[:, None]
    density[:, xi] = (p * np.exp(-0.5 * ((yy - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))).sum(1)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Setting 3 — Mode splitting across three regimes", fontsize=13)
cmap_r = {0: "steelblue", 1: "darkorange", 2: "seagreen"}
cmap_k = plt.cm.tab10

# Panel 1: training scatter coloured by regime
ax = axes[0, 0]
for r, color, label in [(0, "steelblue", "regime 1: unimodal"),
                        (1, "darkorange", "regime 2: bimodal"),
                        (2, "seagreen",  "regime 3: trimodal")]:
    mask = regime.numpy() == r
    ax.scatter(x_train.numpy()[mask], y_train.numpy()[mask],
               s=2, alpha=0.25, color=color, label=label)
ax.axvline(-0.5, color="k", lw=1, ls="--", alpha=0.5)
ax.axvline( 0.5, color="k", lw=1, ls="--", alpha=0.5)
ax.set_title("Training data (coloured by regime)"); ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(markerscale=4, fontsize=8)

# Panel 2: learned density heatmap
ax = axes[0, 1]
im = ax.imshow(density, origin="lower", aspect="auto",
               extent=[-1.5, 1.5, -2.5, 2.5], cmap="magma", interpolation="bilinear")
fig.colorbar(im, ax=ax, label="p(y | x)")
ax.axvline(-0.5, color="white", lw=1, ls="--", alpha=0.7)
ax.axvline( 0.5, color="white", lw=1, ls="--", alpha=0.7)
ax.set_title("Learned density  p(y | x)"); ax.set_xlabel("x"); ax.set_ylabel("y")

# Panel 3: component means
ax = axes[1, 0]
for k in range(NUM_COMPONENTS):
    ax.plot(x_vis_np, mu_vis[:, k], color=cmap_k(k), label=f"μ_{k}")
ax.axvline(-0.5, color="k", lw=1, ls="--", alpha=0.4)
ax.axvline( 0.5, color="k", lw=1, ls="--", alpha=0.4)
ax.set_title("Component means  μₖ(x)"); ax.set_xlabel("x"); ax.set_ylabel("μₖ"); ax.legend(fontsize=8)

# Panel 4: mixing weights
ax = axes[1, 1]
for k in range(NUM_COMPONENTS):
    ax.plot(x_vis_np, pi_vis[:, k], color=cmap_k(k), label=f"π_{k}")
ax.axvline(-0.5, color="k", lw=1, ls="--", alpha=0.4)
ax.axvline( 0.5, color="k", lw=1, ls="--", alpha=0.4)
ax.set_ylim(0, 1)
ax.set_title("Mixing weights  πₖ(x)"); ax.set_xlabel("x"); ax.set_ylabel("πₖ"); ax.legend(fontsize=8)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "03_mode_splitting.png")
plt.savefig(out, dpi=150)
print(f"Figure saved → {out}")
try:
    plt.show()
except Exception:
    pass
