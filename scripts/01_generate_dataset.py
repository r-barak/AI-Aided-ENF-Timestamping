#!/usr/bin/env python3
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from src.config import get_args
from src.data.processing import process_directory_parallel
from src.data.enf_extraction import extract_enf_parallel
from src.data.filtering import gt_alignment
from src.data.triplet_mine import generate_data
from src.constants import DIR_RAW, DIR_INTERIM, DIR_FEATURES, SUBDIR_QUERIES, SUBDIR_REFS

def main():
    args = get_args()
    
    # print("=== Stage 1: Signal Preprocessing (Split & Resample) ===")
    # Input/Output directory structure based on WHU dataset
    query_input_dir = args.train_data_dir / DIR_RAW / SUBDIR_QUERIES
    reference_input_dir = args.train_data_dir / DIR_RAW / SUBDIR_REFS
    query_split_dir = args.train_data_dir / DIR_INTERIM / SUBDIR_QUERIES
    reference_split_dir = args.train_data_dir / DIR_INTERIM / SUBDIR_REFS

    # Process queries (sliced into various lengths) and whole reference database recordings
    # process_directory_parallel(
    #     input_dir=query_input_dir,
    #     output_dir=query_split_dir,
    #     is_reference=False,
    #     target_fs=args.fs,
    #     segment_durations=args.segment_dur,
    #     overlap_min=args.overlap_min,
    #     supported_ext=args.supported_ext
    # )
    # process_directory_parallel(
    #     input_dir=reference_input_dir,
    #     output_dir=reference_split_dir,
    #     is_reference=True,
    #     target_fs=args.fs,
    #     segment_durations=args.segment_dur,
    #     overlap_min=args.overlap_min,
    #     supported_ext=args.supported_ext
    # )

    # print("\n=== Stage 2: Feature Extraction (Multi-Harmonic ENF) ===")
    query_enf_dir = args.train_data_dir / DIR_FEATURES / SUBDIR_QUERIES
    reference_enf_dir = args.train_data_dir / DIR_FEATURES / SUBDIR_REFS

    # extract_enf_parallel(
    #     input_dir=query_split_dir, 
    #     output_dir=query_enf_dir, 
    #     is_reference=False, 
    #     args=args
    # )
    # extract_enf_parallel(
    #     input_dir=reference_split_dir, 
    #     output_dir=reference_enf_dir, 
    #     is_reference=True, 
    #     args=args
    # )

    # print("\n=== Stage 3: Ground Truth Alignment Verification ===")
    analysis_csv_path = args.train_data_dir / "time_stamp_analysis.csv"

    # rows = gt_alignment(
    #     query_enf_dir=query_enf_dir,
    #     reference_enf_dir=reference_enf_dir,
    #     harmonics_freq=args.harmonics_freq, 
    #     allowed_lag_sec=args.allowed_lag_sec
    # )

    # df = pd.DataFrame(rows)
    # df.to_csv(analysis_csv_path, index=False)

    print("\n=== Stage 4: Contrastive Dataset Construction (Triplet Mining) ===")
    df = pd.read_csv(analysis_csv_path)

    train_pairs, val_pairs = generate_data(query_enf_dir=query_enf_dir, 
                      reference_enf_dir=reference_enf_dir, 
                      df=df, 
                      corr_metric=args.corr_metric, 
                      query_harmonic=args.train_query_harmonic, 
                      train_set_percentage=args.train_percentage, 
                      harmonics_freq=args.harmonics_freq, 
                      allowed_lag_sec=args.allowed_lag_sec)


    torch.save(train_pairs, args.train_data_dir / "train_pairs.pt")
    torch.save(val_pairs, args.train_data_dir / "val_pairs.pt")
    print(f"\n--> Successfully built dataset! Pt vectors written to: {args.train_data_dir}")

if __name__ == "__main__":
    main()