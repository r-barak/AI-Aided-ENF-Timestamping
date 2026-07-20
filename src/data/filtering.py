import numpy as np
from tqdm import tqdm
from scipy.signal import fftconvolve
from collections import defaultdict

from src.evaluation.metrics import moving_average

def _sliding_sum(x, m):
    """Computes the sum of elements inside a sliding window of size m."""
    c = np.cumsum(np.pad(x, (1, 0), mode='constant'))
    return c[m:] - c[:-m]

def calc_cc_array(q, r, eps=1e-12):
    """Computes sliding-window Cross-Correlation coefficients between query and reference."""
    N = len(q)
    sum_xy = fftconvolve(r, q[::-1], mode='valid')
    sum_y = _sliding_sum(r, N)
    sum_y2 = _sliding_sum(r**2, N)
    sum_x = np.sum(q)
    sum_x2 = np.sum(q**2)
    
    numerator = (N * sum_xy) - (sum_x * sum_y)
    var_x = (N * sum_x2) - (sum_x**2)
    var_y = (N * sum_y2) - (sum_y**2)
    var_y = np.maximum(var_y, 0)
    denom = np.sqrt(var_x) * np.sqrt(var_y)

    return numerator / (denom + eps)

def calc_nm_array(q, r, eps=1e-12):
    """Computes sliding-window Normalized Minimum Squared Distance coefficients."""
    N = len(q)
    sum_xy = fftconvolve(r, q[::-1], mode='valid')
    sum_y2 = _sliding_sum(r**2, N)
    sum_x2 = np.sum(q**2)
    
    numerator = sum_x2 + sum_y2 - (2 * sum_xy)
    return numerator / (sum_y2 + eps)

def time_stamping(query, ref, eps=1e-12):
    q = moving_average(query, 20).astype(np.float32)
    r = moving_average(ref, 20).astype(np.float32)
    return {
        'cc': np.argmax(calc_cc_array(q, r)),
        'nm': np.argmin(calc_nm_array(q, r))
    }

def gt_alignment(query_enf_dir, reference_enf_dir, harmonics_freq, allowed_lag_sec):
    enf_subdirs = [f"{f}hz" for f in harmonics_freq]

    print("Loading reference signals into memory...")
    ref_signals = defaultdict(list)
    for subdir in enf_subdirs:
        for path in sorted((reference_enf_dir / subdir).rglob('*.npz')):
            ref_name = path.stem.split('_')[0]
            with np.load(path) as data:
                ref_signals[ref_name].append(data['f'].copy())

    base_query_dir = query_enf_dir / enf_subdirs[0]
    query_paths = list(base_query_dir.rglob('*.npz'))

    rows = []

    with tqdm(total=len(query_paths), desc="Verifying Labeled Ground Truths") as pbar:
        for base_path in query_paths:
            fname_parts = base_path.stem.split('_')
            ref_name = fname_parts[0]
            split_name = f"{fname_parts[1]}_{fname_parts[2]}"
            gt_time = int(fname_parts[-2])

            rel_path = base_path.relative_to(base_query_dir)
            ref_arrays = ref_signals[ref_name]

            row_data = {'experiment_name': ref_name, 'split_name': split_name}

            # Iterate through Query Frequencies (i)
            for i, subdir in enumerate(enf_subdirs):
                query_path = query_enf_dir / subdir / rel_path
                
                with np.load(query_path) as query_data:
                    query_f = query_data['f']
                
                # Capture the recording length on the very first pass only
                if i == 0:
                    row_data['recording_length'] = len(query_f)
                
                # Iterate through Reference Frequencies (j)
                for j, ref_f in enumerate(ref_arrays):
                    timestamps = time_stamping(query_f, ref_f)
                    
                    for metric, timestamp in timestamps.items():
                        key = f"{metric}_q{harmonics_freq[i]}_r{harmonics_freq[j]}"
                        ts_sec = timestamp // 2  # Downsample time resolution factor
                        row_data[key] = ts_sec if abs(gt_time - ts_sec) < allowed_lag_sec else None
                            
            rows.append(row_data)
            pbar.update(1)
    
    return rows