import torch

def calc_nm_score(targets, refs, eps=1e-12):
    sum_t2 = (targets * targets).sum(dim=1, keepdim=True)
    sum_r2 = (refs * refs).sum(dim=1, keepdim=True).T
    dots   = targets @ refs.T
    ssd = sum_t2 + sum_r2 - 2 * dots
    nm  = ssd / (sum_r2 + eps)

    return nm

def calc_cc_score(targets, refs, eps=1e-12):
    t = targets - targets.mean(dim=1, keepdim=True)
    r = refs - refs.mean(dim=1, keepdim=True)
    num = t @ r.T
    t_norm = (t.pow(2).sum(dim=1, keepdim=True)).sqrt()
    r_norm = (r.pow(2).sum(dim=1, keepdim=True)).sqrt().T
    denom = t_norm * r_norm

    return num / (denom + eps)

def compute_top_k(k, sim, ref_exp, ref_pos, test_exp, test_pos, time_window=60):
    """
    Compute top-k scores for each sample, so we won't need to insert a very large amount of data to train the model
    """
    topk_idx = sim.topk(k=k, dim=1).indices
    pred_exp = ref_exp[topk_idx]
    pred_pos = ref_pos[topk_idx]

    exp_match = pred_exp.eq(test_exp.unsqueeze(1))
    time_match = (pred_pos - test_pos.unsqueeze(1)).abs().lt(time_window)
    pos_mask = exp_match & time_match # Boolean mask of all positive pairs

    return pos_mask, topk_idx


import numpy as np
import torch
from scipy.signal import fftconvolve

def moving_average(signal, window=20):
    """
    Applies a moving average filter to smooth frequency fluctuation signals.
    Supports both 1D NumPy arrays and 3D PyTorch tensors.
    """
    if isinstance(signal, torch.Tensor):
        B, C, L = signal.shape
        x = signal.cpu().numpy().reshape(B * C, L)
        c = np.cumsum(np.pad(x, ((0, 0), (1, 0)), mode="constant"), axis=1)
        core = (c[:, window:] - c[:, :-window]) / window
        pad_left = (window - 1) // 2
        pad_right = window - 1 - pad_left
        out = np.pad(core, ((0, 0), (pad_left, pad_right)), mode="edge")
        return torch.from_numpy(out).reshape(B, C, L).to(signal.dtype).to(signal.device)
    
    x = signal.astype(np.float32, copy=False)
    if window <= 1:
        return x
    c = np.cumsum(np.pad(x, (1, 0), mode='constant'))
    core = (c[window:] - c[:-window]) / window
    pad_left = (window - 1) // 2
    pad_right = window - 1 - pad_left
    return np.pad(core, (pad_left, pad_right), mode='edge')


def _sliding_sum(x, m):
    """Computes the sum of elements inside a sliding window of size m."""
    c = np.cumsum(np.pad(x, (1, 0), mode='constant'))
    return c[m:] - c[:-m]


def calc_cc_array(t, r, eps=1e-12):
    """Computes sliding-window Cross-Correlation coefficients between target and reference."""
    N = len(t)
    sum_xy = fftconvolve(r, t[::-1], mode='valid')
    sum_y = _sliding_sum(r, N)
    sum_y2 = _sliding_sum(r**2, N)
    sum_x = np.sum(t)
    sum_x2 = np.sum(t**2)
    
    numerator = (N * sum_xy) - (sum_x * sum_y)
    var_x = (N * sum_x2) - (sum_x**2)
    var_y = (N * sum_y2) - (sum_y**2)
    var_y = np.maximum(var_y, 0)
    denom = np.sqrt(var_x) * np.sqrt(var_y)

    return numerator / (denom + eps)


def calc_nm_array(t, r, eps=1e-12):
    """Computes sliding-window Normalized Minimum Squared Distance coefficients."""
    N = len(t)
    sum_xy = fftconvolve(r, t[::-1], mode='valid')
    sum_y2 = _sliding_sum(r**2, N)
    sum_x2 = np.sum(t**2)
    
    numerator = sum_x2 + sum_y2 - (2 * sum_xy)
    return numerator / (sum_y2 + eps)


def calc_cc_score(targets, refs, eps=1e-12):
    """Batched PyTorch implementation of Pearson Cross-Correlation."""
    t = targets - targets.mean(dim=1, keepdim=True)
    r = refs - refs.mean(dim=1, keepdim=True)
    num = t @ r.T
    t_norm = (t.pow(2).sum(dim=1, keepdim=True)).sqrt()
    r_norm = (r.pow(2).sum(dim=1, keepdim=True)).sqrt().T
    denom = t_norm * r_norm
    return num / (denom + eps)


def calc_nm_score(targets, refs, eps=1e-12):
    """
    Batched PyTorch implementation of Normalized Minimum Squared Distance.
    Uses the algebraic expansion equation:
    
    $SSD = \|targets\|^2 + \|refs\|^2 - 2 \cdot (targets \cdot refs^T)$
    """
    sum_t2 = (targets * targets).sum(dim=1, keepdim=True)
    sum_r2 = (refs * refs).sum(dim=1, keepdim=True).T
    dots   = targets @ refs.T
    ssd = sum_t2 + sum_r2 - 2 * dots
    return ssd / (sum_r2 + eps)


def compute_top_k(k, sim, ref_exp, ref_pos, test_exp, test_pos, time_window=60):
    """
    Evaluates whether the true matching reference identity and its corresponding temporal 
    alignment window fall within the Top-K similarity rank.
    """
    topk_idx = sim.topk(k=k, dim=1).indices
    pred_exp = ref_exp[topk_idx]
    pred_pos = ref_pos[topk_idx]

    exp_match = pred_exp.eq(test_exp.unsqueeze(1))
    time_match = (pred_pos - test_pos.unsqueeze(1)).abs().lt(time_window)
    pos_mask = exp_match & time_match 

    return pos_mask, topk_idx