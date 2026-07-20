import numpy as np
import torch
import random
from tqdm import tqdm
from collections import defaultdict

from src.evaluation.metrics import moving_average
from src.data.filtering import calc_cc_array, calc_nm_array


def bucket_samples_by_length(dataset_dict):
    buckets = defaultdict(list)
    for samples in dataset_dict.values():
        for rec_ids, time_indices, signals in samples:
            length = signals.shape[-1]
            buckets[length].append((rec_ids, time_indices, signals))
    return dict(buckets)

def _best_idx_away_from_gt(scores, gt_time, gap, metric):
    n = scores.size
    if n == 0: raise ValueError("Empty correlation array")

    left_bound = max(0, gt_time - gap)
    right_bound = min(n, gt_time + gap + 1)

    # Edge case: The gap covers the entire array
    if left_bound == 0 and right_bound == n:
        return int(np.argmax(scores)) if metric == 'cc' else int(np.argmin(scores))

    best_idx, best_val = 0, -np.inf if metric == 'cc' else np.inf

    # 1. Search Left Side
    if left_bound > 0:
        idx_l = int(np.argmax(scores[:left_bound])) if metric == 'cc' else int(np.argmin(scores[:left_bound]))
        best_idx, best_val = idx_l, scores[idx_l]

    # 2. Search Right Side
    if right_bound < n:
        idx_r = int(np.argmax(scores[right_bound:])) if metric == 'cc' else int(np.argmin(scores[right_bound:]))
        val_r = scores[right_bound + idx_r]
        
        if (metric == 'cc' and val_r > best_val) or (metric == 'nm' and val_r < best_val):
            best_idx = right_bound + idx_r

    return best_idx

def get_positive(query, ref_signals, gt_time, exp_name, num_harmonics):
    gt_time = int(gt_time)
    arr = np.stack(
        [ref_signals[exp_name][i][gt_time:gt_time + len(query)] for i in range(num_harmonics)],
        axis=0
    ).astype(np.float32, copy=False)

    return torch.from_numpy(ref_signals[:, gt_time : gt_time + len(query)])
    
    block = ref_signals[:, gt_time : gt_time + query_len]
    return torch.from_numpy(block)

def get_soft_negative(query, ref_signals, gt_time, exp_name, num_harmonics, metric, gap):
    ts, tensors = [], []
    for k in range(num_harmonics):
        ref_k = ref_signals[exp_name][k]

        scores = calc_cc_array(query, ref_k) if metric == 'cc' else calc_nm_array(query, ref_k)

        i = _best_idx_away_from_gt(scores, gt_time, gap, metric)

        block = np.stack(
            [ref_signals[exp_name][j][i:i + len(query)] for j in range(num_harmonics)],
            axis=0
        ).astype(np.float32, copy=False)
        tensors.append(torch.from_numpy(block))
        ts.append(i)
    return ts, tensors

def get_hard_negative(query, ref_signals, gt_time, exp_name, num_harmonics, metric, allowed_pool, gap):
    out = []
    initial_best = -float('inf') if metric == 'cc' else float('inf')

    valid_candidates = [
        rec_id for rec_id in ref_signals.keys() 
        if rec_id != exp_name and rec_id in allowed_pool
    ]

    for k in range(num_harmonics):
        best_val = initial_best
        best_entry = None

        for rec_id in valid_candidates:
            ref_k = ref_signals[rec_id][k]

            if ref_k.shape[0] <= len(query):
                continue

            scores = calc_cc_array(query, ref_k) if metric == 'cc' else calc_nm_array(query, ref_k)

            i = _best_idx_away_from_gt(scores, gt_time, gap, metric)
            score = float(scores[i])
            
            if (metric == 'cc' and score > best_val) or (metric == 'nm' and score < best_val):
                best_val = score
                best_entry = (rec_id, i)

        if best_entry is None:
            raise ValueError(f"No hard negative found for exp {exp_name} harmonic {k}.")

        choose_rec, i = best_entry
        block = np.stack(
            [ref_signals[choose_rec][j][i:i + len(query)] for j in range(num_harmonics)],
            axis=0
        ).astype(np.float32, copy=False)
        out.append((choose_rec, i, torch.from_numpy(block)))
    return out

def generate_data(query_enf_dir, reference_enf_dir, df, corr_metric, query_harmonic, train_set_percentage, harmonics_freq, allowed_lag_sec):
    enf_subdirs = [f"{f}hz" for f in harmonics_freq]
    query_freq = harmonics_freq[query_harmonic]
    metric_cols = [c for c in df.columns if c.startswith(f"{corr_metric}_q{query_freq}")]
    train_df = df.dropna(subset=metric_cols, how='all')

    unique_exps = list(train_df['experiment_name'].unique())
    random.shuffle(unique_exps)

    # train/validation partitioning based on unique recording identifiers
    split_idx = int(len(unique_exps) * train_set_percentage)
    train_pool_set = [f"{exp_id:03d}" for exp_id in unique_exps[:split_idx]]
    val_pool_set = [f"{exp_id:03d}" for exp_id in unique_exps[split_idx:]]

    # Smooth full references via unified moving average mapping logic
    ref_signals = defaultdict(list)
    for subdir in enf_subdirs:
        for path in sorted((reference_enf_dir / subdir).rglob('*.npz')):
            ref_name = path.stem.split('_')[0]
            with np.load(path) as data:
                ref_signals[ref_name].append(moving_average(data['f'], 20).astype(np.float32))
    
    

    train_dict, val_dict = defaultdict(list), defaultdict(list)

    allowed_gap = allowed_lag_sec * 2
    train_df['gt_time'] = (train_df[metric_cols].mean(axis=1, skipna=True) * 2).astype(int)
    records = train_df[['experiment_name', 'split_name', 'gt_time']].to_dict('records')

    with tqdm(total=len(records), desc="Mining Triplet Pairs") as pbar:
        for row in records:
            exp_name = f"{row['experiment_name']:03d}"
            gt_time = row['gt_time']

            if exp_name in train_pool_set:
                curr_dict, curr_pool = train_dict, train_pool_set
            elif exp_name in val_pool_set:
                curr_dict, curr_pool = val_dict, val_pool_set
            else:
                print("ERROR")
                continue

            file_name = f"{exp_name}_{row['split_name']}.npz"
            with np.load((query_enf_dir / enf_subdirs[query_harmonic] / exp_name / file_name)) as query_data:
                query = moving_average(query_data['f'], window=20).astype(np.float32, copy=False)

            query_tensor = torch.from_numpy(np.stack([query] * len(enf_subdirs), axis=0))
            pos_tensor = get_positive(query, ref_signals, gt_time, exp_name, len(enf_subdirs))
            soft_ts, soft_neg_tensors = get_soft_negative(query, ref_signals, gt_time, exp_name, len(enf_subdirs), corr_metric, allowed_gap)
            hard_neg_tensors = get_hard_negative(query, ref_signals, gt_time, exp_name, len(enf_subdirs), corr_metric, curr_pool, allowed_gap)

            exp_tensor = torch.tensor((
                int(exp_name), 
                int(exp_name), 
                int(exp_name), 
                int(exp_name), 
                int(exp_name),
                int(hard_neg_tensors[0][0]), 
                int(hard_neg_tensors[1][0]), 
                int(hard_neg_tensors[2][0])
            ), dtype=torch.int32)

            start_idx_tensor = torch.tensor((
                gt_time, 
                gt_time, 
                soft_ts[0], 
                soft_ts[1], 
                soft_ts[2],
                hard_neg_tensors[0][1], 
                hard_neg_tensors[1][1], 
                hard_neg_tensors[2][1]
            ), dtype=torch.int32)

            signal_tensor = torch.stack((
                query_tensor,
                pos_tensor,
                soft_neg_tensors[0], 
                soft_neg_tensors[1], 
                soft_neg_tensors[2],
                hard_neg_tensors[0][2], 
                hard_neg_tensors[1][2], 
                hard_neg_tensors[2][2],
            ))

            entry = (exp_tensor, start_idx_tensor, signal_tensor)
            curr_dict[exp_name].append(entry)
            pbar.update(1)

    train_pairs = bucket_samples_by_length(train_dict)
    val_pairs = bucket_samples_by_length(val_dict)

    return train_pairs, val_pairs