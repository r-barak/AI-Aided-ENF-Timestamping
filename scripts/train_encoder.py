#!/usr/bin/env python3
import torch

from src.config import get_args
from src.models.encoder import TwoStageTCNEncoder
from src.training.trainer import train_encoder


def main():
    args = get_args()

    encoder_train_dataset = torch.load(args.train_data_dir / "train_pairs.pt")
    encoder_val_dataset = torch.load(args.train_data_dir / "val_pairs.pt")

    encoder = TwoStageTCNEncoder()
    encoder = train_encoder(model=encoder, 
                            train_dataset=encoder_train_dataset, 
                            val_dataset=encoder_val_dataset, 
                            epochs=args.num_epochs, 
                            batch_size=args.batch_size,
                            lr=args.lr,
                            allowed_lag_sec=args.allowed_lag_sec,
                            device=args.device)
    

if __name__ == "__main__":
    main()