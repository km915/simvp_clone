import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TrafficDataset(Dataset):
    def __init__(self, X, Y):
        super(TrafficDataset, self).__init__()
        self.X = (X + 1) / 2
        self.Y = (Y + 1) / 2
        self.mean = 0
        self.std = 1

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data   = torch.tensor(self.X[index]).float()
        labels = torch.tensor(self.Y[index]).float()
        return data, labels


def load_data(batch_size, val_batch_size, data_root, num_workers, **kwargs):
    taxibj_dir = os.path.join(data_root, 'taxibj')

    # handle both old .npz format and new separate .npy files
    npz_path = os.path.join(taxibj_dir, 'dataset.npz')
    if os.path.exists(npz_path):
        dataset = np.load(npz_path)
        X_train = dataset['X_train']
        Y_train = dataset['Y_train']
        X_test  = dataset['X_test']
        Y_test  = dataset['Y_test']
    else:
        # separate X.npy and Y.npy
        X = np.load(os.path.join(taxibj_dir, 'X.npy'))
        Y = np.load(os.path.join(taxibj_dir, 'Y.npy'))
        # 90/10 split
        split = int(len(X) * 0.9)
        X_train, X_test = X[:split], X[split:]
        Y_train, Y_test = Y[:split], Y[split:]

    train_set = TrafficDataset(X=X_train, Y=Y_train)
    test_set  = TrafficDataset(X=X_test,  Y=Y_test)

    dataloader_train = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers)
    dataloader_test  = DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers)

    return dataloader_train, None, dataloader_test, 0, 1




# import torch
# import numpy as np
# from torch.utils.data import Dataset


# class TrafficDataset(Dataset):
#     def __init__(self, X, Y):
#         super(TrafficDataset, self).__init__()
#         self.X = (X + 1) / 2
#         self.Y = (Y + 1) / 2
#         self.mean = 0
#         self.std = 1

#     def __len__(self):
#         return self.X.shape[0]

#     def __getitem__(self, index):
#         data = torch.tensor(self.X[index, ::]).float()
#         labels = torch.tensor(self.Y[index, ::]).float()
#         return data, labels

# def load_data(
#         batch_size, val_batch_size,
#         data_root, num_workers):

#     dataset = np.load(data_root+'taxibj/dataset.npz')
#     X_train, Y_train, X_test, Y_test = dataset['X_train'], dataset['Y_train'], dataset['X_test'], dataset['Y_test']

#     train_set = TrafficDataset(X=X_train, Y=Y_train)
#     test_set = TrafficDataset(X=X_test, Y=Y_test)

#     dataloader_train = torch.utils.data.DataLoader(
#         train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
#     dataloader_test = torch.utils.data.DataLoader(
#         test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

#     return dataloader_train, None, dataloader_test, 0, 1