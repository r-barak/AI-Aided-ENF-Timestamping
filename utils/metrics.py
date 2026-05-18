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