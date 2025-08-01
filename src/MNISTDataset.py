import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

import os
import json
from PIL import Image

from device import device


class MNISTDataset(Dataset):
    def __init__(self, path : str, train : bool = True, transforms=None):
        super().__init__()
        self.path = os.path.join(path, 'train' if train else 'test')
        self.transforms = transforms

        with open(os.path.join(path, 'format.json'), 'r') as fp:
            self.format = json.load(fp)

        self.files = []
        self.length = 0
        self.targets = torch.eye(n=10, device=device)

        for _dir, _target in self.format.items():
            cur_path = os.path.join(self.path, _dir)
            class_files = os.listdir(cur_path)
            self.length += len(class_files)
            self.files.extend(map(lambda x: (os.path.join(cur_path, x), _target), class_files)) # директория, метка класса
            

    def __getitem__(self, index):
        file_path, target = self.files[index]
        target = self.targets[target]

        image = Image.open(file_path)

        if self.transforms:
            image = self.transforms(image)
        else:
            image = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)])(image)

        return image, target

    def __len__(self):
        return self.length