import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class KTHDataset(Dataset):
    """
    KTH Action Recognition Dataset.
    Folder structure expected:
        data_root/kth/boxing/*.avi
        data_root/kth/handclapping/*.avi
        data_root/kth/handwaving/*.avi
        data_root/kth/jogging/*.avi
        data_root/kth/running/*.avi
        data_root/kth/walking/*.avi

    Following SimVP paper: persons 1-16 for train, 17-25 for test.
    Input: 10 frames, Output: 20 frames (or 40).
    Image size: 128x128, grayscale.
    """

    ACTIONS = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

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

        # persons 1-16 train, 17-25 test — following SimVP paper
        if is_train:
            self.persons = list(range(1, 17))
        else:
            self.persons = list(range(17, 26))

        self.clips = self._build_clips()

    def _build_clips(self):
        """
        Read all videos, extract all valid clips of length n_frames_total.
        Returns list of numpy arrays of shape (n_frames_total, H, W).
        """
        clips = []
        for action in self.ACTIONS:
            action_dir = os.path.join(self.root, action)
            if not os.path.exists(action_dir):
                print(f'Warning: {action_dir} not found, skipping.')
                continue

            for fname in sorted(os.listdir(action_dir)):
                if not fname.endswith('.avi'):
                    continue

                # KTH filenames: person01_boxing_d1_uncomp.avi
                # extract person number
                try:
                    person_id = int(fname.split('_')[0].replace('person', ''))
                except:
                    continue

                if person_id not in self.persons:
                    continue

                video_path = os.path.join(action_dir, fname)
                frames = self._read_video(video_path)

                if frames is None or len(frames) < self.n_frames_total:
                    continue

                # extract all valid clips with stride 1
                for start in range(0, len(frames) - self.n_frames_total + 1,
                                   self.n_frames_total):
                    clip = frames[start: start + self.n_frames_total]
                    clips.append(clip)

        print(f'KTH {"train" if self.is_train else "test"}: {len(clips)} clips')
        return clips

    def _read_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # convert to grayscale, resize to image_size x image_size
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (self.image_size, self.image_size))
            frames.append(frame)
        cap.release()
        return np.array(frames, dtype=np.float32) if frames else None

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]  # (n_frames_total, H, W)

        # normalize to [0, 1]
        clip = clip / 255.0

        # add channel dim: (n_frames_total, 1, H, W)
        clip = clip[:, np.newaxis, :, :]

        input_frames  = torch.from_numpy(clip[:self.n_frames_input]).float()
        output_frames = torch.from_numpy(clip[self.n_frames_input:]).float()

        return input_frames, output_frames


def load_data(batch_size, val_batch_size, data_root, num_workers,
              n_frames_output=20, **kwargs):

    train_set = KTHDataset(root=data_root, is_train=True,
                           n_frames_input=10, n_frames_output=n_frames_output)
    test_set  = KTHDataset(root=data_root, is_train=False,
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