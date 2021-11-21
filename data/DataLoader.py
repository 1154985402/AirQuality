import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torchvision import transforms

class Datalder(Dataset):
    def __init__(self, data_pth, train=True, qt="so2"):
        super(Datalder, self).__init__()
        self.data = self.get_data(data_pth)
        self.train = train
        self.qt = qt

    def __getitem__(self, item):
        dataX = self.data[item, 0:15]
        if self.qt == "so2":
            datay = self.data[item, 15]
        else:
            datay = self.data[item, 16]
        return dataX, datay

    def __len__(self):
        return len(self.data)

    def get_data(self, data_pth):
        data_pd = pd.read_csv(data_pth)
        feature = list(data_pd.columns)
        print(feature)
        for i in range(10):
            data_pd = data_pd.ffill()
        data_pd = np.array(data_pd, dtype=np.float32)
        data_pd = (data_pd-data_pd.mean())/data_pd.std()
        return data_pd


