"""
Setting 4 — 2D input: quadrant-dependent conditional distributions.

True generative process
-----------------------
    (x₁, x₂) ~ Uniform(-1, 1)²
    r = √(x₁² + x₂²)   (distance from origin)
    θ = atan2(x₂, x₁)  (angle)

    The output distribution depends on which quadrant the input lies in:

    Q1  x₁ > 0, x₂ > 0   →  0.8·N(+2, 0.2²) + 0.2·N(-1, 0.2²)   (right-skewed bimodal)
    Q2  x₁ < 0, x₂ > 0   →  0.8·N(-2, 0.2²) + 0.2·N(+1, 0.2²)   (left-skewed bimodal)
    Q3  x₁ < 0, x₂ < 0   →  0.5·N(-1.5,0.2²) + 0.5·N(+1.5,0.2²) (symmetric bimodal)
    Q4  x₁ > 0, x₂ < 0   →  N(0, (0.3 + 0.6·r)²)                 (unimodal, grows with r)

    The encoder must disentangle two input dimensions to infer the right mixture.
    A 1D encoder cannot solve this.

What to look for
----------------
- Mean heatmap: four distinct quadrant patterns
- Std heatmap: Q4 shows radius-dependent uncertainty; other quadrants are tight
- Per-quadrant density curves: visibly different shapes in each corner
- Model needs ≥ 2 components; Q4 just needs 1

Run
---
    uv run python ablations/04_2d_input.py
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

N_TRAIN       = 8_000
N_EPOCHS      = 4_000
LR            = 3e-3
NUM_COMPONENTS = 6
HIDDEN        = 128
SEED          = 2

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

x1 = torch.empty(N_TRAIN).uniform_(-1.0, 1.0)
x2 = torch.empty(N_TRAIN).uniform_(-1.0, 1.0)
r  = (x1 ** 2 + x2 ** 2).sqrt()

q1 = (x1 > 0) & (x2 > 0)
q2 = (x1 < 0) & (x2 > 0)
q3 = (x1 < 0) & (x2 < 0)
q4 = (x1 > 0) & (x2 < 0)

noise = torch.randn(N_TRAIN)
u     = torch.rand(N_TRAIN)

y_q1 = torch.where(u < 0.8, 2.0 + 0.2 * noise, -1.0 + 0.2 * noise)
y_q2 = torch.where(u < 0.8, -2.0 + 0.2 * noise, 1.0 + 0.2 * noise)
y_q3 = torch.where(u < 0.5, -1.5 + 0.2 * noise,  1.5 + 0.2 * noise)
y_q4 = 0.0 + (0.3 + 0.6 * r) * noise           # unimodal, heteroscedastic

y_train = torch.where(q1, y_q1,
          torch.where(q2, y_q2,
          torch.where(q3, y_q3, y_q4)))

quadrant = torch.where(q1, torch.zeros_like(x1),
           torch.where(q2, torch.ones_like(x1),
           torch.where(q3, 2 * torch.ones_like(x1),
                           3 * torch.ones_like(x1)))).long()

x_input = torch.stack([x1, x2], dim=1)   # (N, 2)

# ---------------------------------------------------------------------------
# Model  — encoder takes 2D input
# ---------------------------------------------------------------------------

encoder = nn.Sequential(
    nn.Linear(2, HIDDEN), nn.Tanh(),
    nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
    nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),   # extra depth for 2D
)
decoder   = MDNDecoder(embedding_size=HIDDEN, mlp_hidden_size=HIDDEN, num_components=NUM_COMPONENTS)
criterion = MDNCriterion()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

print(f"Training MDN ({NUM_COMPONENTS} components, 2D input) for {N_EPOCHS} epochs …")
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
# Evaluation on an (x₁, x₂) grid
# ---------------------------------------------------------------------------

encoder.eval(); decoder.eval()

G = 80   # grid resolution
x1_g = torch.linspace(-1, 1, G)
x2_g = torch.linspace(-1, 1, G)
X1, X2 = torch.meshgrid(x1_g, x2_g, indexing="ij")   # (G, G)
grid_input = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)  # (G², 2)

with torch.no_grad():
    d_grid: GMMDistribution = decoder(encoder(grid_input))

mean_grid = d_grid.mean.reshape(G, G).numpy()          # E[y | x₁, x₂]
std_grid  = d_grid.variance.sqrt().reshape(G, G).numpy()

# Per-quadrant representative conditional distributions
representatives = {
    "Q1 (+,+)": torch.tensor([[0.6,  0.6]]),
    "Q2 (−,+)": torch.tensor([[-0.6,  0.6]]),
    "Q3 (−,−)": torch.tensor([[-0.6, -0.6]]),
    "Q4 (+,−)": torch.tensor([[0.6, -0.6]]),
}
y_query = np.linspace(-3.5, 3.5, 400)

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(14, 10))
fig.suptitle("Setting 4 — 2D input: quadrant-dependent conditional distributions", fontsize=13)
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

extent = [-1, 1, -1, 1]

# Panel 1: E[y | x₁, x₂]
ax = fig.add_subplot(gs[0, 0])
im1 = ax.imshow(mean_grid.T, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r")
fig.colorbar(im1, ax=ax, label="E[y]")
ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.4)
ax.set_title("Predicted mean  E[y | x₁, x₂]"); ax.set_xlabel("x₁"); ax.set_ylabel("x₂")

# Panel 2: Std[y | x₁, x₂]
ax = fig.add_subplot(gs[0, 1])
im2 = ax.imshow(std_grid.T, origin="lower", aspect="auto", extent=extent, cmap="YlOrRd")
fig.colorbar(im2, ax=ax, label="Std[y]")
ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.4)
ax.set_title("Predicted std  √Var[y | x₁, x₂]"); ax.set_xlabel("x₁"); ax.set_ylabel("x₂")

# Panel 3: training scatter (x₁, x₂) coloured by y value
ax = fig.add_subplot(gs[0, 2])
sc = ax.scatter(x1.numpy(), x2.numpy(), c=y_train.numpy(),
                cmap="RdBu_r", s=2, alpha=0.3, vmin=-2.5, vmax=2.5)
fig.colorbar(sc, ax=ax, label="y")
ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.4)
ax.set_title("Training data (coloured by y)"); ax.set_xlabel("x₁"); ax.set_ylabel("x₂")

# Panels 4–7 (bottom row, span all 3 cols): per-quadrant p(y | x) density curves
ax = fig.add_subplot(gs[1, :])
y_tensor = torch.tensor(y_query, dtype=torch.float32)

with torch.no_grad():
    for (label, pt), color in zip(representatives.items(), colors):
        d_pt: GMMDistribution = decoder(encoder(pt))
        pi_k    = d_pt.pi.squeeze(0).numpy()      # (K,)
        mu_k    = d_pt.mu.squeeze(0).numpy()
        sigma_k = d_pt.sigma.squeeze(0).numpy()

        yy = y_query[:, None]
        pdf = (pi_k * np.exp(-0.5 * ((yy - mu_k) / sigma_k) ** 2)
               / (sigma_k * np.sqrt(2 * np.pi))).sum(1)
        ax.plot(y_query, pdf, color=color, lw=2, label=label)

ax.set_title("Conditional distributions p(y | x) at four representative points")
ax.set_xlabel("y"); ax.set_ylabel("p(y | x)")
ax.legend(fontsize=10)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "04_2d_input.png")
plt.savefig(out, dpi=150)
print(f"Figure saved → {out}")
try:
    plt.show()
except Exception:
    pass
