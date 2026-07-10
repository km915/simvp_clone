import os
import cv2
import random as pyrandom
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class KITTIDataset(Dataset):
    def __init__(self, root, is_train=True,
                 n_frames_input=10, n_frames_output=1,
                 image_size=(128, 160),
                 irregular=False, g_max=4, seed=0):
        super(KITTIDataset, self).__init__()
        self.output_root = os.path.join(root, 'kitti', 'Output')
        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = n_frames_input + n_frames_output
        self.image_size = image_size
        self.mean = 0
        self.std = 1

        self.irregular = irregular
        self.g_max = g_max
        self.seed = seed

        self.samples = self._build_index()
        print(f'KITTI {"train" if is_train else "test"}: '
              f'{len(self.samples)} clips indexed '
              f'({"irregular" if irregular else "regular"})')

    def _worst_case_span(self):
        return (self.n_frames_input - 1) * self.g_max + self.n_frames_output + 1

    def _build_index(self):
        drive_folders = sorted([
            os.path.join(self.output_root, d)
            for d in os.listdir(self.output_root)
            if os.path.isdir(os.path.join(self.output_root, d))
        ])

        split_idx = max(1, int(len(drive_folders) * 0.9))
        drive_folders = drive_folders[split_idx:] if self.is_train else drive_folders[:split_idx]

        samples = []
        for folder in drive_folders:
            jpg_files = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith('.jpg')
            ])

            if not self.irregular:
                if len(jpg_files) < self.n_frames_total:
                    continue
                for start in range(0, len(jpg_files) - self.n_frames_total + 1,
                                   self.n_frames_total):
                    samples.append((jpg_files, start, len(jpg_files)))
            else:
                min_required = self._worst_case_span()
                if len(jpg_files) < min_required:
                    continue
                n_windows = max(1, len(jpg_files) // self.n_frames_total)
                for _ in range(n_windows):
                    samples.append((jpg_files, None, len(jpg_files)))

        return samples

    def _draw_gaps_and_start(self, n_available, rng):
        gaps = [rng.randint(1, self.g_max) for _ in range(self.n_frames_input - 1)]
        total_needed = sum(gaps) + self.n_frames_output + 1
        max_start = n_available - total_needed
        start = rng.randint(0, max_start) if max_start > 0 else 0
        return gaps, start

    def _read_indices(self, jpg_files, indices):
        H, W = self.image_size
        frames = []
        for i in indices:
            img = cv2.imread(jpg_files[i])
            if img is None:
                frames.append(frames[-1] if frames else np.zeros((H, W, 3), dtype=np.float32))
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (W, H))
            frames.append(img.astype(np.float32))
        return np.array(frames)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        jpg_files, fixed_start, n_available = self.samples[idx]

        if not self.irregular:
            indices = list(range(fixed_start, fixed_start + self.n_frames_total))
            ts_vec = np.ones(self.n_frames_input - 1, dtype=np.float32)
        else:
            rng = pyrandom.Random() if self.is_train else pyrandom.Random(self.seed + idx)
            gaps, start = self._draw_gaps_and_start(n_available, rng)
            idxs = [start]
            for g in gaps:
                idxs.append(idxs[-1] + g)
            last_input = idxs[-1]
            for i in range(1, self.n_frames_output + 1):
                idxs.append(last_input + i)
            indices = idxs
            ts_vec = np.array(gaps, dtype=np.float32)

        clip = self._read_indices(jpg_files, indices)
        clip = clip / 255.0
        clip = clip.transpose(0, 3, 1, 2)

        input_frames = torch.from_numpy(clip[:self.n_frames_input]).float()
        output_frames = torch.from_numpy(clip[self.n_frames_input:]).float()
        ts_vec = torch.from_numpy(ts_vec).float()

        return input_frames, ts_vec, output_frames


def load_data(batch_size, val_batch_size, data_root,
              num_workers, n_frames_output=1,
              irregular=False, g_max=4, **kwargs):

    train_set = KITTIDataset(root=data_root, is_train=True,
                             n_frames_input=10, n_frames_output=n_frames_output,
                             irregular=irregular, g_max=g_max)
    test_set = KITTIDataset(root=data_root, is_train=False,
                            n_frames_input=10, n_frames_output=n_frames_output,
                            irregular=irregular, g_max=g_max)

    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                   pin_memory=True, num_workers=num_workers)
    dataloader_val = DataLoader(test_set, batch_size=val_batch_size, shuffle=False,
                                 pin_memory=True, num_workers=num_workers)
    dataloader_test = DataLoader(test_set, batch_size=val_batch_size, shuffle=False,
                                  pin_memory=True, num_workers=num_workers)

    return dataloader_train, dataloader_val, dataloader_test, 0, 1