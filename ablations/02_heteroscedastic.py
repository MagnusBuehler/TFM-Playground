"""
Setting 2 — Heteroscedastic noise with x-dependent weights and spreads.

True generative process
-----------------------
    x ~ Uniform(-2, 2)
    y | x is a 3-component mixture whose weights AND spreads vary with x:

    Mode A  mean = sin(πx)           std = 0.1 + 0.4·(x/2)²     weight ∝ e^(+x)
    Mode B  mean = -sin(πx)          std = 0.15                  weight ∝ e^(-x)
    Mode C  mean = 1.5·cos(πx/2)     std = 0.3                   weight = 0.2 (fixed)

    Normalised so weights sum to 1. Mode A dominates for large x,
    mode B dominates for small x, mode C is a persistent background.

A fixed-variance Gaussian decoder misses both the mode structure and the
heteroscedastic spread that widens toward the edges.

What to look for
----------------
- Density heatmap should show the sin/cos oscillation and widening spread
- Component spreads σ_k(x) should be clearly non-constant for the learned mode A
- Mixing weights π_k(x) should cross over near x = 0

Run
---
    uv run python ablations/02_heteroscedastic.py
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

N_TRAIN   = 6_000
N_EPOCHS  = 3_000
LR        = 3e-3
NUM_COMPONENTS = 6
HIDDEN    = 128
SEED      = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

x_train = torch.empty(N_TRAIN).uniform_(-2.0, 2.0)

# x-dependent mixing weights (unnormalised)
w_a = torch.exp( x_train)         # grows rightward
w_b = torch.exp(-x_train)         # grows leftward
w_c = torch.full_like(x_train, 0.4 * (w_a + w_b).mean().item())  # fixed background
z   = w_a + w_b + w_c
pi_a, pi_b, pi_c = w_a / z, w_b / z, w_c / z

# Sample component indices
u = torch.rand(N_TRAIN)
component = torch.where(u < pi_a, torch.zeros_like(u),
            torch.where(u < pi_a + pi_b, torch.ones_like(u),
                        2 * torch.ones_like(u))).long()

std_a = 0.1 + 0.4 * (x_train / 2) ** 2   # heteroscedastic: wide at edges
std_b = torch.full_like(x_train, 0.15)
std_c = torch.full_like(x_train, 0.30)

mean_a =  torch.sin(torch.pi * x_train)
mean_b = -torch.sin(torch.pi * x_train)
mean_c =  1.5 * torch.cos(torch.pi * x_train / 2)

noise = torch.randn(N_TRAIN)
y_train = torch.where(component == 0, mean_a + std_a * noise,
          torch.where(component == 1, mean_b + std_b * noise,
                                      mean_c + std_c * noise))

x_input = x_train.unsqueeze(-1)   # (N, 1)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

encoder = nn.Sequential(
    nn.Linear(1, HIDDEN), nn.Tanh(),
    nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
)
decoder  = MDNDecoder(embedding_size=HIDDEN, mlp_hidden_size=HIDDEN, num_components=NUM_COMPONENTS)
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

x_vis = torch.linspace(-2.0, 2.0, 300)
with torch.no_grad():
    d_vis: GMMDistribution = decoder(encoder(x_vis.unsqueeze(-1)))

pi_vis    = d_vis.pi.numpy()      # (300, K)
mu_vis    = d_vis.mu.numpy()      # (300, K)
sigma_vis = d_vis.sigma.numpy()   # (300, K)
x_vis_np  = x_vis.numpy()

# Density heatmap
y_vis   = torch.linspace(-4.0, 4.0, 400)
density = np.zeros((400, 300))
for xi in range(300):
    p  = pi_vis[xi];  m = mu_vis[xi];  s = sigma_vis[xi]
    yy = y_vis.numpy()[:, None]
    density[:, xi] = (p * np.exp(-0.5 * ((yy - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))).sum(1)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Setting 2 — Heteroscedastic noise with x-dependent weights and spreads", fontsize=13)
cmap_k = plt.cm.tab10

# Panel 1: training scatter
ax = axes[0, 0]
sc = ax.scatter(x_train.numpy(), y_train.numpy(), c=component.numpy(),
                cmap="tab10", vmin=0, vmax=2, s=2, alpha=0.25)
fig.colorbar(sc, ax=ax, ticks=[0, 1, 2], label="component")
ax.set_title("Training data (coloured by true component)"); ax.set_xlabel("x"); ax.set_ylabel("y")

# Panel 2: learned density
ax = axes[0, 1]
im = ax.imshow(density, origin="lower", aspect="auto",
               extent=[-2, 2, -4, 4], cmap="magma", interpolation="bilinear")
fig.colorbar(im, ax=ax, label="p(y | x)")
ax.set_title("Learned density  p(y | x)"); ax.set_xlabel("x"); ax.set_ylabel("y")

# Panel 3: component spreads σ_k(x)
ax = axes[1, 0]
for k in range(NUM_COMPONENTS):
    ax.plot(x_vis_np, sigma_vis[:, k], color=cmap_k(k), label=f"σ_{k}")
ax.set_title("Component spreads  σₖ(x)"); ax.set_xlabel("x"); ax.set_ylabel("σₖ"); ax.legend(fontsize=8)

# Panel 4: mixing weights π_k(x)
ax = axes[1, 1]
for k in range(NUM_COMPONENTS):
    ax.plot(x_vis_np, pi_vis[:, k], color=cmap_k(k), label=f"π_{k}")
ax.set_ylim(0, 1)
ax.set_title("Mixing weights  πₖ(x)"); ax.set_xlabel("x"); ax.set_ylabel("πₖ"); ax.legend(fontsize=8)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "02_heteroscedastic.png")
plt.savefig(out, dpi=150)
print(f"Figure saved → {out}")
try:
    plt.show()
except Exception:
    pass
