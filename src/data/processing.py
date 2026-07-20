from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import multiprocessing as mp
from tqdm import tqdm


def add_gaussian_noise(signal, snr_db):
    """
    Adds Additive White Gaussian Noise (AWGN) to a signal for a target SNR.
    """
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def _process_file_worker(args_tuple):
    """
    Internal worker function for parallel processing.
    """
    file_path, output_dir, is_reference, target_fs, segment_durations, overlap_min = args_tuple

    try:
        sample_rate, data = wavfile.read(file_path)

        # Convert stereo to mono by averaging channels
        if data.ndim > 1:
            data = data.mean(axis=1).astype(data.dtype)

        # Resample to target frequency
        data_resampled = resample_poly(data, up=target_fs, down=sample_rate)
        data_resampled = data_resampled.astype(np.float32)

        base_name = file_path.stem
        signal_folder = output_dir / base_name
        signal_folder.mkdir(parents=True, exist_ok=True)

        if is_reference:
            out_path = signal_folder / f"{base_name}.npz"
            np.savez_compressed(out_path, signal=data_resampled, sample_rate=target_fs)
        else:
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
        print(f"Failed to process {file_path}: {e}")

def process_directory_parallel(input_dir, output_dir, is_reference, target_fs, segment_durations, overlap_min, supported_ext=".wav"):
    """
    Parallel processing of all audio recording files in a directory.
    """
    input_path, output_path = Path(input_dir), Path(output_dir)

    if isinstance(supported_ext, str):
        ext_tuple = (supported_ext,)
    else:
        ext_tuple = tuple(supported_ext)

    files = [
        f for f in input_path.iterdir() 
        if f.is_file() and f.name.endswith(ext_tuple)
    ]

    args_list = [
        (file_path, output_path, is_reference, target_fs, segment_durations, overlap_min) 
        for file_path in files
    ]

    if not args_list:
        print(f"No {ext_tuple} files found in {input_path}")
        return

    num_cores = max(1, mp.cpu_count() - 1) # Leave 1 CPU core free so your OS doesn't freeze
    with mp.Pool(num_cores) as pool:
        list(tqdm(
            pool.imap_unordered(_process_file_worker, args_list), 
            total=len(args_list), 
            desc=f"Processing {'References' if is_reference else 'Queries'}",
            unit="file"
        ))