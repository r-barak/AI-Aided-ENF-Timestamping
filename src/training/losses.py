import torch
import torch.nn.functional as F

def get_batch_relations(rec_ids, time_indices, anchors, B, allowed_lag_sec, device):
    # Create an [N, N] matrix where entry (i,j) is True if from the same recording
    same_rec = rec_ids.unsqueeze(1) == rec_ids.unsqueeze(0)  
    same_rec_per_anchor = same_rec[anchors] # Shape: [B, N]
    
    # Exclude an anchor from being its own positive pair
    same_rec_per_anchor[torch.arange(B, device=device), anchors] = False

    # Positive pairs: same recording AND within the acceptable time window tolerance
    anchor_time = time_indices[anchors]
    time_diff = (time_indices.unsqueeze(0) - anchor_time.unsqueeze(1)).abs()
    positives_per_anchor = same_rec_per_anchor & (time_diff < (allowed_lag_sec * 2))

    # Negative pairs: everything else (excluding the anchor itself)
    negatives_per_anchor = ~positives_per_anchor
    negatives_per_anchor[torch.arange(B, device=device), anchors] = False

    return positives_per_anchor, negatives_per_anchor


def info_nce(sim, pos_mask, temperature):
    logits = sim / temperature

    # Denominator: logsumexp of similarities between anchors and all samples in the batch
    denom = torch.logsumexp(logits, dim=1)

    # Numerator: logsumexp of similarities between anchors and positive samples
    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
    numer = torch.logsumexp(pos_logits, dim=1)
    
    return (denom - numer).mean()


def triplet_inbatch_hardest_mine(sim, pos_mask, neg_mask, margin):
    # Hardest negative: maximum similarity score among negative samples
    sim_only_neg = sim.masked_fill(~neg_mask, float("-inf"))
    sim_hard_neg = sim_only_neg.max(dim=1).values

    # Worst positive: minimum similarity score among positive samples
    sim_only_pos = sim.masked_fill(~pos_mask, float("inf"))
    sim_worst_pos = sim_only_pos.min(dim=1).values

    return F.relu(margin + sim_hard_neg - sim_worst_pos).mean()


def compute_encoder_loss(embeddings, rec_ids, time_indices, batch_size, allowed_lag_sec,
                         lambda_1=1.0, lambda_2=1.0, margin=0.2, temperature=0.07):
    N = len(rec_ids)
    step = N // batch_size
    device = rec_ids.device

    # L2 normalize embeddings to compute stable cosine similarities
    norm_embeddings = F.normalize(embeddings, dim=1)
    sim = norm_embeddings @ norm_embeddings.t() # Matrix shape: [N, N]
    
    # Set diagonal to -inf to exclude self-similarity from positive candidates
    sim.fill_diagonal_(float("-inf"))
    
    # Select anchors for similarity matrix computation
    anchors = torch.arange(0, N, step, device=device)       
    sim_per_anchor = sim[anchors]

    pos_mask, neg_mask = get_batch_relations(rec_ids, time_indices, anchors, batch_size, allowed_lag_sec, device)

    loss_nce = info_nce(sim_per_anchor, pos_mask, temperature)
    loss_triplet = triplet_inbatch_hardest_mine(sim_per_anchor, pos_mask, neg_mask, margin)

    return (lambda_1 * loss_nce) + (lambda_2 * loss_triplet)


def listwise_ce(scores, pos_mask, temperature):
    """
    Computes Listwise Cross-Entropy Loss for training fusion module.
    """
    logits = scores / temperature
    log_den = torch.logsumexp(logits, dim=-1)
    
    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
    log_num = torch.logsumexp(pos_logits, dim=-1)
    
    return -(log_num - log_den).mean()