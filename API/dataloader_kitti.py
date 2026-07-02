import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class KITTIDataset(Dataset):
    """
    KITTI Raw Dataset for video prediction.
    Expected structure:
        data_root/kitti/2011_09_26_drive_0001_sync/image_02/data/0000000000.png
        data_root/kitti/2011_09_26_drive_0001_sync/image_02/data/0000000001.png
        ...

    Extracts clips of (n_frames_input + n_frames_output) consecutive frames.
    Image size: 128x160 (H x W), RGB -> 3 channels.
    Uses 90% of drives for train, 10% for test.
    """

    def __init__(self, root, is_train=True,
                 n_frames_input=10, n_frames_output=1,
                 image_size=(128, 160)):
        super(KITTIDataset, self).__init__()
        self.root = os.path.join(root, 'kitti')
        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = n_frames_input + n_frames_output
        self.image_size = image_size  # (H, W)
        self.mean = 0
        self.std = 1

        self.clips = self._build_clips()

    def _find_image_dirs(self):
        """Find all image_02/data directories under kitti root."""
        image_dirs = []
        for drive in sorted(os.listdir(self.root)):
            drive_path = os.path.join(self.root, drive)
            if not os.path.isdir(drive_path):
                continue
            img_dir = os.path.join(drive_path, 'image_02', 'data')
            if os.path.exists(img_dir):
                image_dirs.append(img_dir)
        return image_dirs

    def _build_clips(self):
        image_dirs = self._find_image_dirs()

        # 90/10 train/test split by drive
        split_idx = max(1, int(len(image_dirs) * 0.9))
        if self.is_train:
            image_dirs = image_dirs[:split_idx]
        else:
            image_dirs = image_dirs[split_idx:]

        clips = []
        for img_dir in image_dirs:
            frames = self._read_frames(img_dir)
            if frames is None or len(frames) < self.n_frames_total:
                continue

            for start in range(0, len(frames) - self.n_frames_total + 1,
                               self.n_frames_total):
                clip = frames[start: start + self.n_frames_total]
                clips.append(clip)

        print(f'KITTI {"train" if self.is_train else "test"}: {len(clips)} clips')
        return clips

    def _read_frames(self, img_dir):
        files = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith('.png') or f.endswith('.jpg')
        ])
        if not files:
            return None

        frames = []
        H, W = self.image_size
        for fname in files:
            img = cv2.imread(os.path.join(img_dir, fname))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (W, H))  # cv2 takes (W, H)
            frames.append(img.astype(np.float32))

        return np.array(frames) if frames else None  # (N, H, W, 3)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]  # (n_frames_total, H, W, 3)

        # normalize to [0, 1]
        clip = clip / 255.0

        # transpose to (n_frames_total, 3, H, W)
        clip = clip.transpose(0, 3, 1, 2)

        input_frames  = torch.from_numpy(clip[:self.n_frames_input]).float()
        output_frames = torch.from_numpy(clip[self.n_frames_input:]).float()

        return input_frames, output_frames


def load_data(batch_size, val_batch_size, data_root, num_workers,
              n_frames_output=1, **kwargs):

    train_set = KITTIDataset(root=data_root, is_train=True,
                             n_frames_input=10, n_frames_output=n_frames_output)
    test_set  = KITTIDataset(root=data_root, is_train=False,
                             n_frames_input=10, n_frames_output=n_frames_output)

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