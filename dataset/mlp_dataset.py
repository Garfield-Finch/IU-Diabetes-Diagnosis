import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


class LUPIDataset(Dataset):
    def __init__(self, data_train, data_LUPI, labels):
        self.data_train = torch.FloatTensor(data_train)
        self.data_LUPI = torch.FloatTensor(data_LUPI)
        self.labels = torch.LongTensor(labels)

    def __getitem__(self, index):
        return self.data_train[index], self.data_LUPI[index], self.labels[index]

    def __len__(self):
        return len(self.data_train)
