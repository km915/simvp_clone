import os
import gzip
import random
import numpy as np
import torch
import torch.utils.data as data


def load_mnist(root):
    path = os.path.join(root, 'moving_mnist/train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist


class MovingMNIST(data.Dataset):
    def __init__(self, root, is_train=True, n_frames_input=10, n_frames_output=10,
                 num_objects=[2], transform=None,
                 irregular=False, g_max=4, substeps_per_tick=10, seed=0):
        super(MovingMNIST, self).__init__()

        self.mnist = load_mnist(root)  # always needed now (irregular test can't use fixed file)
        self.length = int(1e4)

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = n_frames_input + n_frames_output
        self.transform = transform

        self.irregular = irregular
        self.g_max = g_max
        self.substeps_per_tick = substeps_per_tick
        self.seed = seed

        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1

        self.mean = 0
        self.std = 1

    def get_random_trajectory(self, seq_length, rng, step_length):
        canvas_size = self.image_size_ - self.digit_size_
        x = rng.random()
        y = rng.random()
        theta = rng.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            y += v_y * step_length
            x += v_x * step_length
            if x <= 0: x, v_x = 0, -v_x
            if x >= 1.0: x, v_x = 1.0, -v_x
            if y <= 0: y, v_y = 0, -v_y
            if y >= 1.0: y, v_y = 1.0, -v_y
            start_y[i] = y
            start_x[i] = x

        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits, rng):
        data = np.zeros((self.n_frames_total, self.image_size_,
                         self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            start_y, start_x = self.get_random_trajectory(
                self.n_frames_total, rng, self.step_length_)
            ind = rng.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames_total):
                top, left = start_y[i], start_x[i]
                bottom, right = top + self.digit_size_, left + self.digit_size_
                data[i, top:bottom, left:right] = np.maximum(
                    data[i, top:bottom, left:right], digit_image)
        return data[..., np.newaxis]

    def generate_moving_mnist_irregular(self, num_digits, rng):
        gaps = [rng.randint(1, self.g_max) for _ in range(self.n_frames_input - 1)]
        spt = self.substeps_per_tick
        substep_gaps = [g * spt for g in gaps]
        total_substeps = sum(substep_gaps) + self.n_frames_output * spt
        fine_step_length = self.step_length_ / spt

        canvas = np.zeros((total_substeps + 1, self.image_size_,
                           self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            start_y, start_x = self.get_random_trajectory(
                total_substeps + 1, rng, fine_step_length)
            ind = rng.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(total_substeps + 1):
                top, left = start_y[i], start_x[i]
                bottom, right = top + self.digit_size_, left + self.digit_size_
                canvas[i, top:bottom, left:right] = np.maximum(
                    canvas[i, top:bottom, left:right], digit_image)

        indices = [0]
        for sg in substep_gaps:
            indices.append(indices[-1] + sg)
        last_input_idx = indices[-1]
        for i in range(1, self.n_frames_output + 1):
            indices.append(last_input_idx + i * spt)

        frames = canvas[indices][..., np.newaxis]
        ts_vec = np.array(gaps, dtype=np.float32)  # real tick-units, incl. multi-tick gaps
        return frames, ts_vec

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output

        if not self.irregular:
            rng = random  # module-level, standard behavior preserved
            num_digits = rng.choice(self.num_objects)
            images = self.generate_moving_mnist(num_digits, rng)
            ts_vec = np.ones(self.n_frames_input - 1, dtype=np.float32)
        else:
            rng = random.Random() if self.is_train else random.Random(self.seed + idx)
            num_digits = rng.choice(self.num_objects)
            images, ts_vec = self.generate_moving_mnist_irregular(num_digits, rng)

        r = 1
        w = int(64 / r)
        images = images.reshape((length, w, r, w, r)).transpose(
            0, 2, 4, 1, 3).reshape((length, r * r, w, w))

        input_ = images[:self.n_frames_input]
        output = images[self.n_frames_input:length] if self.n_frames_output > 0 else []

        output = torch.from_numpy(output / 255.0).contiguous().float()
        input_ = torch.from_numpy(input_ / 255.0).contiguous().float()
        ts_vec = torch.from_numpy(ts_vec).float()

        return input_, ts_vec, output

    def __len__(self):
        return self.length


def load_data(batch_size, val_batch_size, data_root, num_workers,
              n_frames_output=10, irregular=False, g_max=4, **kwargs):

    train_set = MovingMNIST(root=data_root, is_train=True,
                            n_frames_input=10, n_frames_output=n_frames_output, num_objects=[2],
                            irregular=irregular, g_max=g_max)
    test_set = MovingMNIST(root=data_root, is_train=False,
                           n_frames_input=10, n_frames_output=n_frames_output, num_objects=[2],
                           irregular=irregular, g_max=g_max)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_validation = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std