import argparse
from pathlib import Path
import torch

from src.constants import TRAIN_DATASET_NAME, TEST_DATASET_NAME, DIR_FEATURES, SUBDIR_REFS, SUBDIR_QUERIES

def get_args():
    parser = argparse.ArgumentParser(
        description="AI-Aided ENF Timestamping Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ==========================================
    # System & Directory Paths
    # ==========================================
    parser.add_argument('--project_dir', type=str, default='.', 
                        help='Base project directory')

    parser.add_argument('--train_dataset_name', type=str, default=TRAIN_DATASET_NAME, 
                        help='Name of the training dataset folder')

    parser.add_argument('--test_dataset_name', type=str, default=TEST_DATASET_NAME, 
                        help='Name of the testing dataset folder')
    
    # ==========================================
    # Signal Processing & ENF Extraction
    # ==========================================
    parser.add_argument('--network_freq', type=int, default=50, 
                        help='Base electrical network frequency (Hz)')

    parser.add_argument('--harmonics', type=int, nargs='+', default=[1, 2, 3], 
                        help='Harmonics to extract (multipliers of network_freq)')

    parser.add_argument('--fs', type=int, default=400, 
                        help='Target sampling rate after downsampling')

    parser.add_argument('--f_range', type=float, default=0.2, 
                        help='Frequency range around the center frequency')

    parser.add_argument('--window_size_sec', type=int, default=5, 
                        help='Window size in seconds for STFT')
                        
    parser.add_argument('--overlap_ratio', type=float, default=0.9, 
                        help='Overlap ratio for STFT')
                        
    parser.add_argument('--window_type', type=str, default='hamming', 
                        help='Window function type')

    parser.add_argument('--band_pass_order', type=int, default=1501, 
                        help='Order of the FIR bandpass filter')

    # ==========================================
    # Dataset Generation & Triplet Mining
    # ==========================================
    parser.add_argument('--train_percentage', type=float, default=-.8,
                        help='Train size (%) from the full dataset')
    
    parser.add_argument('--segment_dur', type=float, nargs='+', 
                        default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], 
                        help='Durations (in minutes) to split the target signals')

    parser.add_argument('--overlap_min', type=float, default=1.0, 
                        help='Overlap (in minutes) when splitting signals')

    parser.add_argument('--allowed_lag_sec', type=int, default=15, 
                        help='Allowed lag in seconds for ground truth matching')

    parser.add_argument('--corr_metric', type=str, default='cc', choices=['cc', 'nm'],
                        help='Correlation metric used for alignment (cc or nm)')

    parser.add_argument('--train_query_harmonic', type=int, default=1, 
                        help='Target harmonic index used during training dataset generation')

    parser.add_argument('--snr_levels', type=int, nargs='+', default=[-20, -10, 0, 10, 20], 
                        help='SNR levels (dB) for evaluating AWGN robustness')

    parser.add_argument('--supported_ext', type=str, default='.wav', 
                        help='Supported audio file extension')
    
    # ==========================================
    # Training
    # ==========================================
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size')
    
    parser.add_argument('--num_epochs', type=int, default=20, 
                        help='Number of epochs')
    
    parser.add_argument('--lr', type=int, default=1e-3, 
                        help='Learning rate of optimizer')
    # ==========================================
    # Hardware
    # ==========================================
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Disables CUDA training even if available')

    args = parser.parse_args()

    # --- Post-Processing & Dynamic Path Resolution ---
    
    # Device configuration
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    # Path resolutions
    base_dir = Path(args.project_dir).resolve()
    args.train_data_dir = base_dir / 'data' / 'datasets' / args.train_dataset_name
    # args.test_data_dir = base_dir / 'data' / 'datasets' / args.test_dataset_name
    
    # args.train_enf_dir = args.train_data_dir / "enf_data"
    # args.test_enf_dir = args.test_data_dir / "enf_data"
    
    # Derived mathematical variables
    args.harmonics_freq = [h * args.network_freq for h in args.harmonics]

    return args

if __name__ == "__main__":
    # Quick test to verify it works when running the file directly
    config = get_args()
    print(f"Device set to: {config.device}")
    print(f"Training data path: {config.train_data_dir}")
    print(f"Tracking harmonics: {config.harmonics_freq} Hz")



