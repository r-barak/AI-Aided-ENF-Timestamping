import multiprocessing as mp
from pathlib import Path
import numpy as np
from tqdm import tqdm
from scipy.signal import get_window, firwin, filtfilt, spectrogram

def _get_harmonic_filter_args(harmonic, f_center, fs, f_range, window_size_sec, overlap_ratio, window_type, band_pass_order):
    """
    Generates the necessary parameters and FIR filter coefficients for a specific harmonic.
    """
    window_size = int(window_size_sec * fs)
    h_args = {
        'harmonic': harmonic,
        'f_center': f_center,
        'window': get_window(window_type, window_size),
        'noverlap': int(overlap_ratio * window_size),
        'nfft': 100 * window_size,
        'cutoff': [
            (f_center - f_range / 2) / (fs / 2), 
            (f_center + f_range / 2) / (fs / 2)
        ]
    }
    h_args['filter_coeff'] = firwin(
        band_pass_order, 
        h_args['cutoff'], 
        pass_zero=False, 
        window=window_type
    )
    return h_args

def _extract_enf(signal, fs, f_range, h_args):
    """
    Extracts the Electrical Network Frequency (ENF) fluctuation signal 
    from a raw audio signal using STFT and peak tracking.
    """
    # Bandpass filter
    signal = filtfilt(h_args['filter_coeff'], [1], signal)

    # Compute STFT
    f, t, s = spectrogram(
        signal,
        fs=fs,
        window=h_args['window'],
        noverlap=h_args['noverlap'],
        nfft=h_args['nfft'],
        mode='complex'
    )

    # Define frequency range of interest
    f_low = h_args['f_center'] - f_range * h_args['harmonic'] / 2
    f_high = h_args['f_center'] + f_range * h_args['harmonic'] / 2

    # Find indices corresponding to the frequency range
    f1_idx = np.argmin(np.abs(f - f_low))
    f2_idx = np.argmin(np.abs(f - f_high))

    # Select frequency range and corresponding spectrogram values
    f_cut = f[f1_idx:f2_idx + 1]
    s_cut = np.abs(s[f1_idx:f2_idx + 1, :])

    # 4. Find peak frequency corresponding to maximum STFT magnitude
    max_idx = np.argmax(s_cut, axis=0)
    
    # Return deviation from center frequency
    return f_cut[max_idx] - h_args['f_center']

def _extract_enf_worker(args_tuple):
    """
    Internal worker function for parallel processing.
    """
    file_path, input_dir, output_dir, is_reference, f_center, fs, f_range, h_args = args_tuple

    try:
        # Resolve output path based on file type
        if is_reference:
            output_path = output_dir / f"{f_center}hz" / file_path.relative_to(input_dir).parent / file_path.name
        else:
            fname_split = file_path.stem.split('_')
            # Standardize filename back to index representations in seconds
            new_fname = f"{fname_split[0]}_{int(int(fname_split[1])/fs)}_{int(int(fname_split[2])/fs)}.npz"
            output_path = output_dir / f"{f_center}hz" / file_path.relative_to(input_dir).parent / new_fname
        
        # Skip if file was already processed (Allows safe resuming)
        if output_path.exists():
            return
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with np.load(str(file_path)) as data:
            signal = data['signal']
            
        signal_enf = _extract_enf(signal, fs, f_range, h_args)
        np.savez_compressed(str(output_path), f=signal_enf)

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")


def extract_enf_parallel(input_dir, output_dir, is_reference, args):
    """
    Parallel processing wrapper for iterating over files and extracting ENF.
    Assumes `args` contains: harmonics, network_freq, fs, f_range, 
    window_size_sec, overlap_ratio, window_type, band_pass_order.
    """
    input_path, output_path = Path(input_dir), Path(output_dir)

    files = [path for path in input_path.rglob('*.npz') if path.is_file()]
    
    if not files:
        print(f"No .npz files found in {input_path}")
        return

    num_cores = max(1, mp.cpu_count() // 2) #max(1, mp.cpu_count() - 1) # Leave 1 CPU core free so the OS doesn't freeze

    # Process each harmonic independently
    for harmonic in args.harmonics:
        f_center = harmonic * args.network_freq
        
        # Precompute the heavy filter logic once per harmonic
        h_args = _get_harmonic_filter_args(
            harmonic=harmonic, f_center=f_center, fs=args.fs, f_range=args.f_range,
            window_size_sec=args.window_size_sec, overlap_ratio=args.overlap_ratio,
            window_type=args.window_type, band_pass_order=args.band_pass_order
        )

        # Pack arguments for the multiprocessing map
        args_list = [
            (file_path, input_path, output_path, is_reference, f_center, args.fs, args.f_range, h_args) 
            for file_path in files
        ]

        with mp.Pool(num_cores) as pool:
            list(tqdm(
                pool.imap_unordered(_extract_enf_worker, args_list), 
                total=len(args_list), 
                desc=f"Extracting {'References' if is_reference else 'Queries'} ENF at {f_center}Hz",
                unit="file"
            ))