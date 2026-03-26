import h5py
import random
import torch
import numpy as np

from pfns.bar_distribution import get_bucket_limits

def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_default_device():
    device = 'cpu'
    if torch.backends.mps.is_available(): device = 'mps'
    if torch.cuda.is_available(): device = 'cuda'
    return device

def make_global_bucket_edges(filename, n_buckets=100, device=get_default_device(), max_y=5_000_000):
    with h5py.File(filename, "r") as f:
        y = f["y"]
        num_tables, num_datapoints = y.shape

        num_tables_to_use = min(num_tables, max_y // num_datapoints)

        y_subset = np.array(y[:num_tables_to_use, :], dtype=np.float32)
        y_means = y_subset.mean(axis=1, keepdims=True)
        y_stds = y_subset.std(axis=1, keepdims=True, ddof=1) + 1e-8
        ys_concat = ((y_subset - y_means) / y_stds).ravel()

    if ys_concat.size < n_buckets:
        raise ValueError(f"Too few target samples ({ys_concat.size}) to compute {n_buckets} buckets.")

    ys_tensor = torch.tensor(ys_concat, dtype=torch.float32, device=device)
    global_bucket_edges = get_bucket_limits(n_buckets, ys=ys_tensor).to(device)
    return global_bucket_edges


def make_global_bucket_edges_from_prior(prior, n_buckets=100, n_batches=50, device=get_default_device()):
    """Compute global bucket edges by sampling batches from a prior data loader.

    Normalizes each dataset's targets (mean/std) before pooling, matching the
    same per-dataset normalization applied during training.

    Args:
        prior: any iterable yielding dicts with a 'y' key (batch_size, seq_len)
        n_buckets (int): number of quantile buckets
        n_batches (int): how many batches to sample for computing the edges
        device: target device for the returned bucket edges
    Returns:
        Tensor of shape (n_buckets + 1,) with bucket boundary values
    """
    ys = []
    for i, batch in enumerate(prior):
        if i >= n_batches:
            break
        y = batch["y"].float().cpu()
        mean = y.mean(dim=1, keepdim=True)
        std = y.std(dim=1, keepdim=True) + 1e-8
        ys.append(((y - mean) / std).ravel())
    ys_concat = torch.cat(ys)
    if ys_concat.numel() < n_buckets:
        raise ValueError(f"Too few target samples ({ys_concat.numel()}) to compute {n_buckets} buckets.")
    return get_bucket_limits(n_buckets, ys=ys_concat).to(device)
