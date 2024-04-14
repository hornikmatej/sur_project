from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from typing import Tuple, Callable
from torch.utils.data import Dataset
from PIL import ImageFile

from src.dataset import ImageDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

def cutout(x: Image.Image, h: int, w: int, c: int = 3) -> Image.Image:
    """
    Cutout data augmentation. Randomly cuts h by w hole in the image, and fill the whole with zeros.
    # https://arxiv.org/abs/1708.04552
    """
    image_h, image_w = x.size[0], x.size[1]
    x0 = torch.randint(0, image_h + 1 - h, ())
    y0 = torch.randint(0, image_w + 1 - w, ())
    img_array = np.array(x)
    if c == 1:
        img_array[x0:x0+h, y0:y0+w] = 0
    else:
        img_array[x0:x0+h, y0:y0+w, :] = 0

    # Convert the NumPy array back to a PIL image
    return Image.fromarray(img_array)

def calc_normvals(dataset: ImageDataset) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Calculate mean and standard deviation of the dataset.
    """
    # transform list of PIL images to a single NumPy array
    dataset.data = np.stack([np.array(x) for x in dataset.data], axis=0)
    mean = np.mean(dataset.data, axis=(0, 1, 2)) / 255
    std = np.std(dataset.data, axis=(0, 1, 2)) / 255
    print(f'Mean: {mean}, Std: {std}')
    return mean, std

class Config(object):
    """
    Configuration class.
    """
    MAX_LR: float = 0.001
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 0.000125
    GRAD_CLIP: float = 0.1
    NESTEROV: bool = False
    NUM_CLASSES: int = 2
    EPOCHS: int = 30
    BATCH_SIZE: int = 16
    DATA_DIR: str = './data'
    CHECKPOINTS_PATH: Callable = lambda epoch, m_name: f'checkpoints/img_model-{m_name}-{epoch:03d}.pkl'
    NORM_VALS: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = calc_normvals(ImageDataset(DATA_DIR, train=True))
    TRANSFORM_TEST: transforms.Compose = transforms.Compose([
                        transforms.ToTensor(),
                        #transforms.Normalize(*NORM_VALS, inplace=True)
                    ])
    TRANSFORM_TRAIN: transforms.Compose = transforms.Compose([
                        transforms.RandomCrop(80, padding=8, padding_mode='reflect'),
                        transforms.Lambda(lambda x: cutout(x, 20, 20)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        #transforms.Normalize(*NORM_VALS, inplace=True)
                    ])