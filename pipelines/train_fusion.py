

# Loss Functions
def listwise_ce(scores, pos_mask, temperature):
    logits = scores / temperature
    log_den = torch.logsumexp(logits, dim=-1)
    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
    log_num = torch.logsumexp(pos_logits, dim=-1)
    return -(log_num - log_den).mean()

# Validation Loop
@torch.no_grad()
def evaluate(model, loader, temperature, device):
    total_loss = 0
    model.eval()
    for batch in loader:
        scores = batch[0].to(device)
        pos_mask = batch[1].to(device)
        scores = model(scores)
        loss = listwise_ce(scores, pos_mask, temperature)
        total_loss += loss.item()
    return total_loss / len(loader)

# Training Loop
def train_fusion_length_bucketed(train_data, val_data, batch_size, epochs, temperature, lr, weight_decay, drop_last, device="cuda", seed=47):

    dl_tr = ProjectionBucketedBatchLoader(train_data, batch_size=batch_size, drop_last=drop_last, seed=seed)
    dl_va = ProjectionBucketedBatchLoader(val_data, batch_size=batch_size, drop_last=drop_last, seed=seed)

    for batch in dl_tr:
      in_dim = batch[0].shape[-1]
      break
    model = LinearFusion(in_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = {"val_loss": np.inf, "state": None}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dl_tr:
            phi = batch[0].to(device)
            pos_mask = batch[1].to(device)
            scores = model(phi)
            loss = listwise_ce(scores, pos_mask, temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        val_loss = evaluate(model, dl_va, temperature, device=device)
        print(f"Epoch {epoch + 1:02d} | train_loss {total_loss / len(dl_tr):.4f} | val_loss {val_loss:.4f}")

        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"Loaded best model (val_loss={best['val_loss']:.4f})")

    return model