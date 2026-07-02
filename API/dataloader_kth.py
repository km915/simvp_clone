import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class KTHDataset(Dataset):
    """
    KTH Action Recognition Dataset — lazy loading version.
    Stores only (video_path, start_frame) pairs in memory.
    Reads frames on demand in __getitem__.
    """

    ACTIONS = ['boxing', 'handclapping', 'handwaving',
               'jogging', 'running', 'walking']

    def __init__(self, root, is_train=True,
                 n_frames_input=10, n_frames_output=20,
                 image_size=128):
        super(KTHDataset, self).__init__()
        self.root = os.path.join(root, 'kth')
        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = n_frames_input + n_frames_output
        self.image_size = image_size
        self.mean = 0
        self.std = 1

        # persons 1-16 train, 17-25 test
        self.persons = list(range(1, 17)) if is_train else list(range(17, 26))

        # store (video_path, start_frame_index) — no pixel data in memory
        self.samples = self._build_index()
        print(f'KTH {"train" if is_train else "test"}: '
              f'{len(self.samples)} clips indexed (lazy loading)')

    def _build_index(self):
        """
        Walk all videos, count their frames, store clip indices.
        Does NOT read pixel data.
        """
        samples = []
        for action in self.ACTIONS:
            action_dir = os.path.join(self.root, action)
            if not os.path.exists(action_dir):
                continue

            for fname in sorted(os.listdir(action_dir)):
                if not fname.endswith('.avi'):
                    continue
                try:
                    person_id = int(fname.split('_')[0].replace('person', ''))
                except:
                    continue
                if person_id not in self.persons:
                    continue

                video_path = os.path.join(action_dir, fname)

                # only open video to count frames — no pixel reads
                cap = cv2.VideoCapture(video_path)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                if n_frames < self.n_frames_total:
                    continue

                # store one index entry per valid clip
                for start in range(0, n_frames - self.n_frames_total + 1,
                                   self.n_frames_total):
                    samples.append((video_path, start))

        return samples

    def _read_clip(self, video_path, start):
        """Read exactly n_frames_total frames starting at `start`."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames = []
        for _ in range(self.n_frames_total):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            frames.append(frame.astype(np.float32))
        cap.release()

        # pad with last frame if video ended early
        while len(frames) < self.n_frames_total:
            frames.append(frames[-1] if frames else
                          np.zeros((self.image_size, self.image_size),
                                   dtype=np.float32))

        return np.array(frames)  # (n_frames_total, H, W)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, start = self.samples[idx]
        clip = self._read_clip(video_path, start)  # (T, H, W)

        clip = clip / 255.0
        clip = clip[:, np.newaxis, :, :]  # (T, 1, H, W)

        input_frames  = torch.from_numpy(
            clip[:self.n_frames_input]).float()
        output_frames = torch.from_numpy(
            clip[self.n_frames_input:]).float()

        return input_frames, output_frames


def load_data(batch_size, val_batch_size, data_root,
              num_workers, n_frames_output=20, **kwargs):

    train_set = KTHDataset(root=data_root, is_train=True,
                           n_frames_input=10,
                           n_frames_output=n_frames_output)
    test_set  = KTHDataset(root=data_root, is_train=False,
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