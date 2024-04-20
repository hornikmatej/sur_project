from torch.utils.data import Dataset

import os
from PIL import Image

from typing import Any

class ImageDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.data: Any = []
        self.targets: Any = []
        self.populate_data()

    def populate_data(self):
        for folder_name in ['target', 'non_target']:
            if self.train:
                folder = os.path.join(self.root_dir, f"{folder_name}_train")
            else:
                folder = os.path.join(self.root_dir, f"{folder_name}_dev")

            for filename in os.listdir(folder):
                if filename.endswith('.png'):
                    file_path = os.path.join(folder, filename)
                    image = Image.open(file_path)
                    self.data.append(image)
                    if folder_name == 'target':
                        self.targets.append(1)
                    else:
                        self.targets.append(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label