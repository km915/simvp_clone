import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class KITTIDataset(Dataset):
    """
    KITTI Dataset — lazy loading, reads frames on demand.
    Structure: data_root/kitti/Output/<drive_folder>/*.jpg
    Splits drives 90/10 for train/test.
    """

    def __init__(self, root, is_train=True,
                 n_frames_input=10, n_frames_output=1,
                 image_size=(128, 160)):
        super(KITTIDataset, self).__init__()
        self.output_root = os.path.join(root, 'kitti', 'Output')
        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = n_frames_input + n_frames_output
        self.image_size = image_size  # (H, W)
        self.mean = 0
        self.std = 1

        self.samples = self._build_index()
        print(f'KITTI {"train" if is_train else "test"}: '
              f'{len(self.samples)} clips indexed')

    def _build_index(self):
        """
        Walk all drive folders, collect sorted jpg paths,
        store (folder_jpg_list, start_idx) per clip.
        No pixel data loaded.
        """
        # collect all drive folders
        drive_folders = sorted([
            os.path.join(self.output_root, d)
            for d in os.listdir(self.output_root)
            if os.path.isdir(os.path.join(self.output_root, d))
        ])

        # 90/10 split by drive folder
        split_idx = max(1, int(len(drive_folders) * 0.9))
        if self.is_train:
            drive_folders = drive_folders[:split_idx]
        else:
            drive_folders = drive_folders[split_idx:]

        samples = []
        for folder in drive_folders:
            # only jpg files, sorted by filename for temporal order
            jpg_files = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith('.jpg')
            ])

            if len(jpg_files) < self.n_frames_total:
                continue  # skip folders too short

            # store non-overlapping clips
            for start in range(0, len(jpg_files) - self.n_frames_total + 1,
                               self.n_frames_total):
                samples.append((jpg_files, start))

        return samples

    def _read_clip(self, jpg_files, start):
        """Read n_frames_total consecutive jpg files starting at start."""
        H, W = self.image_size
        frames = []
        for i in range(start, start + self.n_frames_total):
            img = cv2.imread(jpg_files[i])
            if img is None:
                # use last good frame if read fails
                frames.append(frames[-1] if frames
                              else np.zeros((H, W, 3), dtype=np.float32))
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (W, H))  # cv2 takes (W, H)
            frames.append(img.astype(np.float32))

        return np.array(frames)  # (n_frames_total, H, W, 3)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        jpg_files, start = self.samples[idx]
        clip = self._read_clip(jpg_files, start)  # (T, H, W, 3)

        # normalize to [0, 1] and transpose to (T, 3, H, W)
        clip = clip / 255.0
        clip = clip.transpose(0, 3, 1, 2)

        input_frames  = torch.from_numpy(
            clip[:self.n_frames_input]).float()
        output_frames = torch.from_numpy(
            clip[self.n_frames_input:]).float()

        return input_frames, output_frames


def load_data(batch_size, val_batch_size, data_root,
              num_workers, n_frames_output=1, **kwargs):

    train_set = KITTIDataset(root=data_root, is_train=True,
                             n_frames_input=10,
                             n_frames_output=n_frames_output)
    test_set  = KITTIDataset(root=data_root, is_train=False,
                             n_frames_input=10,
                             n_frames_output=n_frames_output)

    dataloader_train = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers)
    dataloader_val = DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers)
    dataloader_test = DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers)

    return dataloader_train, dataloader_val, dataloader_test, 0, 1