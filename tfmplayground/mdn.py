import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as D


class GMMDistribution:
    """
    A GMM distribution defined by mixing weights π, means μ, and standard deviations σ.

    Supports log-likelihood training, PDF/CDF evaluation, mean prediction, and sampling.
    All operations are differentiable w.r.t. π, μ, σ.

    Args:
        pi:    (..., k) mixing weights (must sum to 1, e.g. output of softmax)
        mu:    (..., k) component means
        sigma: (..., k) component standard deviations (must be positive)
    """

    def __init__(self, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
        self._mixture = D.MixtureSameFamily(
            D.Categorical(probs=pi),
            D.Normal(loc=mu, scale=sigma),
        )

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log-likelihood of x under the GMM.

        Args:
            x: (...) targets with the same batch shape as π/μ/σ
        Returns:
            (...) log-probabilities
        """
        return self._mixture.log_prob(x)

    def nll_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood — minimise this to train.

        Args:
            x: (...) targets
        Returns:
            (...) per-element NLL; call .mean() to get a scalar loss
        """
        return -self.log_prob(x)

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Probability density at x.

        Args:
            x: (...) query points
        Returns:
            (...) density values
        """
        return self.log_prob(x).exp()

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        """
        Cumulative distribution function at x.

        Computed as Σ_k π_k · Φ((x − μ_k) / σ_k), where Φ is the standard normal CDF.

        Args:
            x: (...) query points
        Returns:
            (...) values in [0, 1]
        """
        # x[..., None] broadcasts against the k-dimension of mu/sigma
        component_cdfs = D.Normal(self.mu, self.sigma).cdf(x.unsqueeze(-1))
        return (self.pi * component_cdfs).sum(dim=-1)

    def sample(self, sample_shape: tuple = ()) -> torch.Tensor:
        """
        Draw samples from the mixture.

        Args:
            sample_shape: prefix shape of the output, e.g. (100,) for 100 samples
        Returns:
            (*sample_shape, ...) samples
        """
        return self._mixture.sample(sample_shape)

    @property
    def mean(self) -> torch.Tensor:
        """Expected value of the mixture: Σ_k π_k μ_k. Shape: (...)."""
        return (self.pi * self.mu).sum(dim=-1)

    @property
    def variance(self) -> torch.Tensor:
        """
        Total variance via the law of total variance:
        Var[X] = Σ_k π_k (σ_k² + μ_k²) − (Σ_k π_k μ_k)²
        Shape: (...)
        """
        second_moment = (self.pi * (self.sigma ** 2 + self.mu ** 2)).sum(dim=-1)
        return second_moment - self.mean ** 2

    @property
    def params(self) -> tuple:
        """Returns (pi, mu, sigma) — useful for inspection or custom operations."""
        return self.pi, self.mu, self.sigma


class MDNCriterion:
    """
    Drop-in criterion for MDN training. Wraps GMMDistribution.nll_loss so
    that train() can call criterion(output, targets) uniformly, where output
    is the GMMDistribution returned by MDNDecoder.forward().
    """

    def __call__(self, output: GMMDistribution, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            output:  GMMDistribution produced by the model's MDNDecoder
            targets: (...) normalised regression targets
        Returns:
            (...) per-element NLL; train() will call .mean() on the result
        """
        return output.nll_loss(targets)


class MDNDecoder(nn.Module):
    """
    Mixture Density Network decoder.

    Projects a hidden representation to the parameters of a k-component GMM:
      hidden → MLP → [logits | means | log_scales] → (π, μ, σ) → GMMDistribution

    Drop-in replacement for the project's Decoder class. The input/output shapes are:
      Input:  (..., embedding_size)
      Output: GMMDistribution with batch shape (...)

    Args:
        embedding_size:  dimension of the incoming hidden state
        mlp_hidden_size: width of the intermediate MLP layer
        num_components:  k — number of Gaussian mixture components
    """

    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_components: int):
        super().__init__()
        self.num_components = num_components
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, mlp_hidden_size),
            nn.GELU(),
            nn.Linear(mlp_hidden_size, 3 * num_components),
        )

    def forward(self, hidden: torch.Tensor) -> GMMDistribution:
        """
        Args:
            hidden: (..., embedding_size) latent representation
        Returns:
            GMMDistribution with batch shape (...)
        """
        raw = self.mlp(hidden)                            # (..., 3k)
        logits, means, log_scales = raw.chunk(3, dim=-1)  # each (..., k)
        pi    = F.softmax(logits, dim=-1)                 # mixing weights, sum to 1
        mu    = means                                     # component means, unconstrained
        sigma = F.softplus(log_scales) + 1e-5             # positive std devs, floor at 1e-5
        return GMMDistribution(pi, mu, sigma)
