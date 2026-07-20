import torch
import random

class BucketedBatchLoader:
    """
    A custom DataLoader that groups sequences of the same length into batches,
    preventing the need for excessive padding in time-series contrastive learning.
    """
    def __init__(self, buckets, batch_size, drop_last, seed=47):
        self.buckets = buckets
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        all_batches = []
        random.seed(self.seed + self.epoch)  

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

class ProjectionBucketedBatchLoader:
    """
    DataLoader specifically formatted for the Linear Fusion layer training,
    grouping pre-computed distance scores.
    """
    def __init__(self, data, batch_size, drop_last=True, seed=47):
        self.data = data
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        self.buckets = {L: [] for L in set(data['L'])}
        for i, L in enumerate(data['L']):
            self.buckets[L].append((data['scores'][i], data['pos_mask'][i], data['L'][i]))

    def __iter__(self):
        all_batches = []
        random.seed(self.seed + self.epoch)  

        for length, samples in self.buckets.items():
            samples = samples.copy()
            random.shuffle(samples)

            for i in range(0, len(samples), self.batch_size):
                batch = samples[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    features = torch.stack([scores for scores, *_ in batch], dim=0)
                    masks = torch.stack([mask for _, mask, _ in batch], dim=0)
                    L = torch.tensor([l for *_, l in batch])
                    all_batches.append((features, masks, L))

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