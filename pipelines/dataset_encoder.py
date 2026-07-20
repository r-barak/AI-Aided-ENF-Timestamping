from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from scipy.io import wavfile
from scipy.signal import resample_poly, firwin, spectrogram, filtfilt
from scipy.signal.windows import get_window
import numpy as np
from scipy.signal import convolve as fftconvolve

############################################
# Dataloader
############################################
class BucketedBatchLoader:
    def __init__(self, buckets, batch_size, drop_last=True, seed=47):
        self.buckets = buckets
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        all_batches = []
        random.seed(self.seed + self.epoch)  # Different shuffle each epoch

        for length, samples in self.buckets.items():
            samples = samples.copy()
            random.shuffle(samples)

            for i in range(0, len(samples), self.batch_size):
                batch = samples[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    rec_ids = torch.cat([exp for exp, *_ in batch], dim=0)
                    time_indices = torch.cat([idx for _, idx, _ in batch], dim=0)
                    tensors = torch.cat([tensor for *_, tensor in batch], dim=0)
                    all_batches.append((rec_ids, time_indices, tensors))

        random.shuffle(all_batches)
        self.epoch += 1
        return iter(all_batches)

    def __len__(self):
        total_batches = 0
        for samples in self.buckets.values():
            n = len(samples)
            if self.drop_last:
                total_batches += n // self.batch_size
            else:
                total_batches += (n + self.batch_size - 1) // self.batch_size
        return total_batches

############################################
# Functions for ENF Extraction
############################################
def h_args_parser(harmonic, f_center, args):
    window_size = int(args['window_size_sec'] * args['fs'])
    h_args = {
        'harmonic': harmonic,
        'f_center': f_center,
        'window': get_window(args['window_type'], window_size),
        'noverlap': int(args['overlap_ratio'] * window_size),
        'nfft': 100 * window_size,
        'cutoff': [(f_center - args['f_range'] / 2) / (args['fs'] / 2), (f_center + args['f_range'] / 2) / (args['fs'] / 2)]
    }
    h_args['filter_coeff'] = firwin(args['band_pass_order'], h_args['cutoff'], pass_zero=False, window=args['window_type'])

    return h_args

def extract_enf(signal, args, h_args):
    signal = filtfilt(h_args['filter_coeff'], [1], signal)

    # Compute STFT
    f, t, s = spectrogram(
        signal,
        fs=args['fs'],
        window=h_args['window'],
        noverlap=h_args['noverlap'],
        nfft=h_args['nfft'],
        mode='complex'
    )

    # Define frequency range of interest
    f_low = h_args['f_center'] - args['f_range'] * h_args['harmonic'] / 2
    f_high = h_args['f_center'] + args['f_range'] * h_args['harmonic'] / 2

    # Find indices corresponding to the frequency range
    f1_idx = np.argmin(np.abs(f - f_low))
    f2_idx = np.argmin(np.abs(f - f_high))

    # Select frequency range and corresponding spectrogram values
    f_cut = f[f1_idx:f2_idx + 1]
    s_cut = np.abs(s[f1_idx:f2_idx + 1, :])

    # Find frequency corresponding to maximum STFT magnitude
    max_idx = np.argmax(s_cut, axis=0)
    return f_cut[max_idx] - h_args['f_center']

def moving_average(signal, window=20):
    x = signal.astype(np.float32, copy=False)
    if window <= 1:
        return x

    c = np.cumsum(np.pad(x, (1, 0), mode='constant'))
    core = (c[window:] - c[:-window]) / window
    pad_left = (window - 1) // 2
    pad_right = window - 1 - pad_left
    return np.pad(core, (pad_left, pad_right), mode='edge')

def _sliding_sum(x, m):
    # returns sum of x inside sliding window of size m
    c = np.cumsum(np.pad(x, (1, 0), mode='constant'))
    return c[m:] - c[:-m]

def calc_cc_array(t, r, eps=1e-12):
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
    N = len(t)
    sum_xy = fftconvolve(r, t[::-1], mode='valid')
    sum_y2 = _sliding_sum(r**2, N)
    sum_x2 = np.sum(t**2)
    numerator = sum_x2 + sum_y2 - (2 * sum_xy)

    return numerator / (sum_y2 + eps)

############################################
# 1. Signal Preprocessing
############################################
def process_file(data_args):
    """
    Worker function: Loads audio, converts to mono, resamples, and saves segments.
    """
    file_path, output_dir, is_reference = data_args

    try:
        sample_rate, data = wavfile.read(file_path)

        # Convert stereo to mono by averaging channels
        if data.ndim > 1:
            data = data.mean(axis=1).astype(data.dtype)

        # Resample to 'RESAMPLED_RATE'
        data_resampled = resample_poly(data, up=args['fs'], down=sample_rate)
        data_resampled = data_resampled.astype(np.float32)

        # Prepare output paths
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        signal_folder = os.path.join(output_dir, base_name)
        os.makedirs(signal_folder, exist_ok=True)

        if is_reference:
            # Reference recordings are saved whole (the "Database" to search against)
            out_path = os.path.join(signal_folder, f"{base_name}.npz")
            np.savez_compressed(out_path, signal=data_resampled, sample_rate=args['fs'])
        else:
            # Target recordings (Queries) are split into various lengths
            step = int(args['overlap_min'] * 60 * args['fs'])
            for dur in args['segment_dur']:
                seg_len = int(dur * 60 * args['fs'])
                for start_idx in range(0, len(data_resampled) - seg_len + 1, step):
                    end_idx = start_idx + seg_len
                    out_path = os.path.join(signal_folder, f"{base_name}_{start_idx}_{end_idx}.npz")
                    np.savez_compressed(out_path, signal=data_resampled[start_idx:end_idx], sample_rate=args['fs'])

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def process_file(data_args: tuple) -> None:
    """
    Worker function: Loads audio, converts to mono, resamples, and saves segments.
    """
    (file_path, output_dir, is_reference, target_fs, overlap_min, segment_durations) = data_args
    # file_path, output_dir = Path(file_path), Path(output_dir)

    try:
        sample_rate, data = wavfile.read(file_path)

        # Convert stereo to mono by averaging channels
        if data.ndim > 1:
            data = data.mean(axis=1).astype(data.dtype)

        # Resample from 'sample_rate' to 'target_fs'
        data_resampled = resample_poly(data, up=target_fs, down=sample_rate)
        data_resampled = data_resampled.astype(np.float32)

        base_name = file_path.stem 
        signal_folder = output_dir / base_name
        signal_folder.mkdir(parents=True, exist_ok=True)

        if is_reference:
            # Reference recordings are saved whole
            out_path = signal_folder / f"{base_name}.npz"
            np.savez_compressed(out_path, signal=data_resampled, sample_rate=target_fs)
        else:
            # Target recordings are split incrementally
            step = int(overlap_min * 60 * target_fs)
            
            for dur in segment_durations:
                seg_len = int(dur * 60 * target_fs)
                
                for start_idx in range(0, len(data_resampled) - seg_len + 1, step):
                    end_idx = start_idx + seg_len
                    out_path = signal_folder / f"{base_name}_{start_idx}_{end_idx}.npz"
                    
                    np.savez_compressed(
                        out_path, 
                        signal=data_resampled[start_idx:end_idx], 
                        sample_rate=target_fs
                    )

    except Exception as e:
        print(f"Failed to process {file_path.name}: {e}")

def process_directory_parallel(
    input_dir: str, 
    output_dir: str, 
    is_reference: bool,
    supported_ext: str | tuple,
    target_fs: int,
    overlap_min: float,
    segment_durations: list[float]
) -> None:
    """
    Parallel processing of all audio recording files in a directory.
    """
    input_path, output_path = Path(input_dir), Path(output_dir)

    files = [
        f for f in input_path.iterdir() 
        if f.is_file() and f.name.endswith(supported_ext)
    ]

    args_list = [
        (f, output_path, is_reference, target_fs, overlap_min, segment_durations) 
        for f in files
    ]

    num_cores = max(1, mp.cpu_count() - 1) # Leave 1 CPU core free so your OS doesn't freeze
    with mp.Pool(num_cores) as pool:
        list(tqdm(
            pool.imap_unordered(process_file, args_list), 
            total=len(args_list), 
            desc=f"Processing {'References' if is_reference else 'Queries'}"
        ))

############################################
