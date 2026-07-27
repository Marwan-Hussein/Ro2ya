import glob
import os
import numpy as np
import torch
from augmentations import augment_clip
from torch.utils.data import Dataset, DataLoader


class GesturesDataset(Dataset):
    def __init__(self, processed_dir, target_folders, train=True, max_seq_len=60):
        """
        processed_dir: folder containing <gesture_name>/<clip>.npy (output of run_batch_processing)
        target_folders: e.g. ['back_button', 'forward_button'] — also defines label order
        train: if True, applies augmentation in __getitem__
        max_seq_len: clips longer than this get center-cropped
        """
        self.train = train
        self.max_seq_len = max_seq_len
        self.label_map = {folder: i for i, folder in enumerate(target_folders)}

        self.samples = []  # list of (npy_path, label)
        for folder, label in self.label_map.items():
            folder_path = os.path.join(processed_dir, folder)
            for npy_path in sorted(glob.glob(os.path.join(folder_path, "*.npy"))):
                self.samples.append((npy_path, label))

        if not self.samples:
            raise RuntimeError(f"No .npy files found under {processed_dir} for folders {target_folders}")


        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            features = np.load(path).astype(np.float32)  # (num_frames, feature_dim)

            if self.train:
                features = augment_clip(features)

            if features.shape[0] > self.max_seq_len:
                start = (features.shape[0] - self.max_seq_len) // 2
                features = features[start: start + self.max_seq_len]

            return torch.from_numpy(features), label


        @property
        def feature_dim(self):
            return int(np.load(self.samples[0][0]).shape[1])

        @property
        def num_classes(self):
            return len(self.label_map)



def collate_pad(batch):
    """Pads a batch to its longest clip and returns a mask (True = real frame)."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
    max_len = int(lengths.max().item())
    feature_dim = sequences[0].shape[1]

    padded = torch.zeros(len(sequences), max_len, feature_dim, dtype=torch.float32)
    mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)

    for i, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        padded[i, :seq_len] = seq
        mask[i, :seq_len] = True

    return padded, mask, lengths, torch.tensor(labels, dtype=torch.long)

def make_loaders(processed_dir, target_folders, max_seq_len=60, batch_size=16, val_frac=0.2, seed=42):
    full_train_ds = GesturesDataset(processed_dir, target_folders, train=True, max_seq_len=max_seq_len)
    full_val_ds = GesturesDataset(processed_dir, target_folders, train=False, max_seq_len=max_seq_len)  # no augmentation

    n = len(full_train_ds)
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = torch.utils.data.random_split(range(n), [n_train, n_val], generator=generator)

    train_subset = torch.utils.data.Subset(full_train_ds, train_idx.indices)
    val_subset = torch.utils.data.Subset(full_val_ds, val_idx.indices)  # same files, but no augmentation

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)

    return train_loader, val_loader, full_train_ds.feature_dim, full_train_ds.num_classes
