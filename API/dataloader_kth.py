import os
import cv2
import random as pyrandom
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


class KTHDataset(Dataset):
    ACTIONS = ['boxing', 'handclapping', 'handwaving',
               'jogging', 'running', 'walking']

    def __init__(self, root, is_train=True,
                 n_frames_input=10, n_frames_output=20,
                 image_size=128,
                 irregular=False, g_max=4, seed=0):
        super(KTHDataset, self).__init__()
        self.root = os.path.join(root, 'kth')
        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = n_frames_input + n_frames_output
        self.image_size = image_size
        self.mean = 0
        self.std = 1

        self.irregular = irregular
        self.g_max = g_max
        self.seed = seed  # base seed for deterministic val/test gaps

        self.persons = list(range(1, 17)) if is_train else list(range(17, 26))
        self.samples = self._build_index()
        print(f'KTH {"train" if is_train else "test"}: '
              f'{len(self.samples)} clips indexed '
              f'({"irregular" if irregular else "regular"})')

    def _worst_case_span(self):
        return (self.n_frames_input - 1) * self.g_max + self.n_frames_output + 1

    def _build_index(self):
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
                cap = cv2.VideoCapture(video_path)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                if not self.irregular:
                    if n_frames < self.n_frames_total:
                        continue
                    for start in range(0, n_frames - self.n_frames_total + 1,
                                       self.n_frames_total):
                        samples.append((video_path, start, n_frames))
                else:
                    min_required = self._worst_case_span()
                    if n_frames < min_required:
                        continue  # too short even for irregular sampling
                    n_windows = max(1, n_frames // self.n_frames_total)
                    for _ in range(n_windows):
                        samples.append((video_path, None, n_frames))

        return samples

    def _draw_gaps_and_start(self, n_frames_available, rng):
        gaps = [rng.randint(1, self.g_max) for _ in range(self.n_frames_input - 1)]
        total_needed = sum(gaps) + self.n_frames_output + 1
        max_start = n_frames_available - total_needed
        start = rng.randint(0, max_start) if max_start > 0 else 0
        return gaps, start

    def _read_clip_regular(self, video_path, start):
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
        while len(frames) < self.n_frames_total:
            frames.append(frames[-1] if frames else
                          np.zeros((self.image_size, self.image_size), dtype=np.float32))
        return np.array(frames)

    def _read_clip_irregular(self, video_path, start, gaps):
        indices = [start]
        for g in gaps:
            indices.append(indices[-1] + g)
        last_input_idx = indices[-1]
        for i in range(1, self.n_frames_output + 1):
            indices.append(last_input_idx + i)

        cap = cv2.VideoCapture(video_path)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame_arr = frames[-1] if frames else np.zeros(
                    (self.image_size, self.image_size), dtype=np.float32)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_arr = cv2.resize(frame, (self.image_size, self.image_size)).astype(np.float32)
            frames.append(frame_arr)
        cap.release()
        return np.array(frames)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, fixed_start, n_frames_available = self.samples[idx]

        if not self.irregular:
            clip = self._read_clip_regular(video_path, fixed_start)
            ts_vec = np.ones(self.n_frames_input - 1, dtype=np.float32)
        else:
            rng = pyrandom.Random() if self.is_train else pyrandom.Random(self.seed + idx)
            gaps, start = self._draw_gaps_and_start(n_frames_available, rng)
            clip = self._read_clip_irregular(video_path, start, gaps)
            ts_vec = np.array(gaps, dtype=np.float32)

        clip = clip / 255.0
        clip = clip[:, np.newaxis, :, :]

        input_frames = torch.from_numpy(clip[:self.n_frames_input]).float()
        output_frames = torch.from_numpy(clip[self.n_frames_input:]).float()
        ts_vec = torch.from_numpy(ts_vec).float()

        return input_frames, ts_vec, output_frames


def load_data(batch_size, val_batch_size, data_root,
              num_workers, n_frames_output=20,
              irregular=False, g_max=4, **kwargs):

    train_set = KTHDataset(root=data_root, is_train=True,
                           n_frames_input=10, n_frames_output=n_frames_output,
                           irregular=irregular, g_max=g_max)
    test_set = KTHDataset(root=data_root, is_train=False,
                          n_frames_input=10, n_frames_output=n_frames_output,
                          irregular=irregular, g_max=g_max)

    dataloader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                   pin_memory=True, num_workers=num_workers)
    dataloader_val = DataLoader(test_set, batch_size=val_batch_size, shuffle=False,
                                 pin_memory=True, num_workers=num_workers)
    dataloader_test = DataLoader(test_set, batch_size=val_batch_size, shuffle=False,
                                  pin_memory=True, num_workers=num_workers)

    return dataloader_train, dataloader_val, dataloader_test, 0, 1