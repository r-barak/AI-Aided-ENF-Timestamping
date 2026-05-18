import torch

# Loss Functions
def get_batch_relations(rec_ids, time_indices, anchors, N, B, dev):
    # same-recording matrix and rows for anchors excluding anchors to themselve
    same_rec = rec_ids.unsqueeze(1) == rec_ids.unsqueeze(0)  # [N, N]
    same_rec_per_anchor = same_rec[anchors]                  # [B, N]
    same_rec_per_anchor[torch.arange(B, device=dev), anchors] = False

    # Find all positives per anchor
    anchor_time = time_indices[anchors]
    time_diff = (time_indices.unsqueeze(0) - anchor_time.unsqueeze(1)).abs()
    positives_per_anchor = same_rec_per_anchor & (time_diff < (args['allowed_lag_sec'] * 2))

    # Find all negatives per anchor
    negatives_per_anchor = ~positives_per_anchor
    negatives_per_anchor[torch.arange(B, device=dev), anchors] = False

    return positives_per_anchor, negatives_per_anchor

def info_nce(sim, pos_mask, temperature):
    """
    Numerator = similarity between anchors and positives
    Denominator = similarity between anchors and all samples in the batch
    """
    logits = sim / temperature
    denom = torch.logsumexp(logits, dim=1)
    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
    numer = torch.logsumexp(pos_logits, dim=1)
    return (denom - numer).mean()

def triplet_inbatch_hardest_mine(sim, pos_mask, neg_mask, batch_size, margin):
    # Hardest negative: max similarity among negatives
    sim_only_neg = sim.masked_fill(~neg_mask, float("-inf"))
    sim_hard_neg = sim_only_neg.max(dim=1).values

    # Worst positive: min similarity among positives
    sim_only_pos = sim.masked_fill(~pos_mask, float("inf"))
    sim_worst_pos = sim_only_pos.min(dim=1).values

    return F.relu(margin + sim_hard_neg - sim_worst_pos).mean()

def compute_loss(embeddings, rec_ids, time_indices, batch_size, lambda_1=1.0, lambda_2=1, margin=0.2, temperature=0.07, gamma=1.0, eps=1e-4):
    N = len(rec_ids)
    step = N // batch_size
    dev = rec_ids.device

    # cosine logits
    norm_embeddings = F.normalize(embeddings, dim=1)
    sim = norm_embeddings @ norm_embeddings.t()          # [N, N]
    sim.fill_diagonal_(float("-inf"))                    # Set denominator
    anchors = torch.arange(0, N, step, device=dev)       # [B]
    sim_per_anchor = sim[anchors]

    positives_per_anchor_mask, negatives_per_anchor_mask = get_batch_relations(rec_ids, time_indices, anchors, N, batch_size, dev)

    info_nce_loss = info_nce(sim_per_anchor, positives_per_anchor_mask, temperature)
    triplet_loss = triplet_inbatch_hardest_mine(sim_per_anchor, positives_per_anchor_mask, negatives_per_anchor_mask, batch_size, margin)

    return lambda_1 * info_nce_loss + lambda_2 * triplet_loss

# Validation Loop
@torch.inference_mode()
def validate(model, dataloader, batch_size, device, visualize=False, epoch=None):
    use_amp = torch.cuda.is_available() and ("cuda" in str(device))
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    loss_sum = 0
    model.eval()
    for rec_ids, time_indices, val_batch in dataloader:
        val_batch    = val_batch.to(device, non_blocking=True)
        rec_ids      = rec_ids.to(device, non_blocking=True)
        time_indices = time_indices.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            embeddings = model(val_batch)
            loss = compute_loss(embeddings, rec_ids, time_indices, batch_size)
        loss_sum += loss.item()

    return loss_sum / len(dataloader)
    
# Training Loop
def train_encoder(model, train_dataset, val_dataset, epochs=10, batch_size=32, lr=1e-3, device=device):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    train_loader = BucketedBatchLoader(train_dataset, batch_size=batch_size, drop_last=True)
    val_loader = BucketedBatchLoader(val_dataset, batch_size=batch_size, drop_last=True)

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # AMP: BF16 on L4/Ampere+, FP16 on T4
    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)

    # best-model tracking
    best_val = float('inf')
    best_state = None
    best_epoch = None

    for epoch in range(epochs):
        model.train()
        loss_sum = 0

        for rec_ids, time_indices, batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            rec_ids = rec_ids.to(device, non_blocking=True)
            time_indices = time_indices.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                embeddings = model(batch)
                loss = compute_loss(
                    embeddings, rec_ids, time_indices,
                    batch_size)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_sum += loss.item()

        avg_train_loss = loss_sum / len(train_loader)
        val_loss = validate(model, val_loader, batch_size, device, visualize=False, epoch=epoch+1)
        print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} / Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"Restored best weights from epoch {best_epoch} (val_loss={best_val:.4f})")

    return model