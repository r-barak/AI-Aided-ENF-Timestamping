import torch
import numpy as np
from torch.optim import AdamW
from src.data.dataloader import BucketedBatchLoader, ProjectionBucketedBatchLoader
from src.training.losses import compute_encoder_loss, listwise_ce

@torch.inference_mode()
def evaluate_encoder(model, dataloader, batch_size, allowed_lag_sec, device, use_amp, amp_dtype):
    loss_sum = 0
    model.eval()
    
    for rec_ids, time_indices, val_batch in dataloader:
        val_batch = val_batch.to(device, non_blocking=True)
        rec_ids = rec_ids.to(device, non_blocking=True)
        time_indices = time_indices.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            embeddings = model(val_batch)
            loss = compute_encoder_loss(embeddings, rec_ids, time_indices, batch_size, allowed_lag_sec)
        loss_sum += loss.item()

    return loss_sum / len(dataloader)


def train_encoder(model, train_dataset, val_dataset, epochs, batch_size, lr, allowed_lag_sec, device):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    train_loader = BucketedBatchLoader(train_dataset, batch_size=batch_size, drop_last=True)
    val_loader = BucketedBatchLoader(val_dataset, batch_size=batch_size, drop_last=True)

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype is torch.float16))

    best_val_loss = float('inf')
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
                loss = compute_encoder_loss(embeddings, rec_ids, time_indices, batch_size, allowed_lag_sec)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_sum += loss.item()

        avg_train_loss = loss_sum / len(train_loader)
        val_loss = evaluate_encoder(model, val_loader, batch_size, allowed_lag_sec, device, use_amp, amp_dtype)
        print(f"[Epoch {epoch+1:02d}] Train Loss: {avg_train_loss:.4f} / Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"--> Restored optimal parameters from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    return model


@torch.inference_mode()
def evaluate_fusion(model, loader, temperature, device):
    """
    Computes listwise validation metric updates for Score Fusion optimization steps.
    """
    total_loss = 0
    model.eval()
    for batch in loader:
        scores = batch[0].to(device)
        pos_mask = batch[1].to(device)
        fused_scores = model(scores)
        loss = listwise_ce(fused_scores, pos_mask, temperature)
        total_loss += loss.item()
    return total_loss / len(loader)


def train_fusion_layer(train_data, val_data, batch_size, epochs, temperature, lr, weight_decay,
                       drop_last, fusion_model_class, device):
    train_loader = ProjectionBucketedBatchLoader(train_data, batch_size=batch_size, drop_last=drop_last)
    val_loader   = ProjectionBucketedBatchLoader(val_data, batch_size=batch_size, drop_last=drop_last)

    in_features = next(iter(train_loader))[0].shape[-1]
    model = fusion_model_class(in_features).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = np.inf
    best_state = None
    best_epoch = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            phi = batch[0].to(device)
            pos_mask = batch[1].to(device)
            
            scores = model(phi)
            loss = listwise_ce(scores, pos_mask, temperature)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = evaluate_fusion(model, val_loader, temperature, device)
        print(f"Epoch {epoch + 1:02d} | Train Loss: {total_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"--> Restored optimal parameters from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    return model