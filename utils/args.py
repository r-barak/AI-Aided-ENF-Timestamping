import argparse
import torch
import numpy as np


def args_parser(device):
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument('--data_folder', type=str, default='./LASSO_data',
                        help="main folder of dataset")

    parser.add_argument('--dataset_type', type=str, default='milodo',
                        help="dataset type")

    parser.add_argument('--train_ratio', type=int, default=0.8,
                        help="train dataset ratio - [0 to 1] percentage")

    parser.add_argument('--val_ratio', type=int, default=0.1,
                        help="validation dataset ratio - [0 to 1] percentage")

    parser.add_argument('--l1_ratio', type=int, default=0.1,
                        help="ratio for l1 regularization")

    parser.add_argument('--noise_scale', type=float, default=0.01,
                        help="noise scale")

    parser.add_argument('--d_features', type=int, default=500,
                        help="dim of X")

    parser.add_argument('--m_meas_per_agent', type=int, default=25,
                        help="amount of data per agent")

    parser.add_argument('--num_instances', type=int, default=1024,
                        help="number of instances")


    # Graph arguments
    parser.add_argument('--n_agents', type=int, default=20,
                        help="number of agents")


    # DADMM arguments
    parser.add_argument('--rho', type=float, default=0.1,
                        help="rho hyperparameter - 0.2603")

    parser.add_argument('--alpha', type=float, default=0.125,
                        help="alpha hyperparameter - 0.125")

    parser.add_argument('--eta', type=float, default=0.0867,
                        help="eta hyperparameter - 0.0867")

    parser.add_argument('--tau', type=float, default=0.05,
                        help="tau hyperparameter - 0.1142")

    parser.add_argument('--num_iters', type=int, default=25,
                        help="number of DADMM iterations")


    # learning arguments
    parser.add_argument('--lr', type=float, default=1e-03,
                        help="optimizer learning rate")

    parser.add_argument('--weight_decay', type=float, default=0, help="weight decay")

    parser.add_argument('--lr_scheduler', action='store_true',
                        help="reduce the learning rate when val_acc has stopped improving (increasing)")

    parser.add_argument('--device', type=str, default=device,
                        help="device to use (gpu or cpu)")

    parser.add_argument('--seed', type=float, default=47,
                        help="manual seed for reproducibility")

    parser.add_argument('--train_type', type=str, default='sequential',
                        help="end2end or sequential")

    parser.add_argument('--num_trained_layers', type=int, default=5,
                        help="number of trained layers (iterations) in sequential learning (No. of DADMM iterations)")

    parser.add_argument('--loss_type', type=str, default='multi_iteration',
                        help="end2end or multi_iteration")

    parser.add_argument('--batch_size', type=int, default=128,
                        help="trainset batch size")

    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help="how many batches to accumulate for optimizer step")

    parser.add_argument('--num_epochs', type=int, default=10,
                        help="number of epochs")

    args, unknown = parser.parse_known_args()
    return args

def args_parser(project_dir, device):
    args = {
        'train_data_dir': project_dir / 'data' / 'datasets' / 'ENF_WHU',
        'test_data_dir': project_dir / 'data' / 'datasets' / 'ENF_wave_mark',
        'network_freq': 50,
        'harmonics': [1,2,3],
        'segment_dur': [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        'allowed_lag_sec': 15,
        'corr_metric': 'cc',
        'overlap_min': 1,
        'supported_ext': ".wav",
        'device': device,
        # Parameters for ENF extraction
        'fs': 400,
        'f_range': 0.2,
        'window_size_sec': 5,
        'overlap_ratio': 0.9,
        'window_type': 'hamming',
        'band_pass_order': 1501,
        'snr_levels': [-20, -10, 0, 10, 20],
        'train_target_harmonic': 1
    }
    args['harmonics_freq'] = [h * args['network_freq'] for h in args['harmonics']]
    args['train_enf_dir'] = args['train_data_dir'] / "enf_data"
    args['test_enf_dir'] = args['test_data_dir'] / "enf_data"
    return args


def initializations(args):
    # Reproducibility
    torch.backends.cudnn.deterministic = True   # Forces cuDNN to use only deterministic algorithms
    # torch.backends.cudnn.benchmark = False    # Usually paired with the above line .deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True