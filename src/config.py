from torchvision import transforms
from PIL import Image
import torch
import numpy as np
from typing import Tuple, Callable
from torch.utils.data import Dataset
from PIL import ImageFile
from typing import Tuple

from src.dataset import ImageDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

def cutout(x: Image.Image, h: int, w: int, c: int = 3, p: float = 0.5) -> Image.Image:
    """
    Cutout data augmentation. Randomly cuts h by w hole in the image, and fill the whole with zeros.
    # https://arxiv.org/abs/1708.04552
    """
    if np.random.rand() > p:
        return x
    
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

def gaus_noise(x: Image.Image, mean: Tuple[float, float] = (0.0, 0.2), std: Tuple[float, float] = (0.05, 0.3), p: float = 0.5) -> Image.Image:
    """
    Add Gaussian noise to the image.
    """
    if np.random.rand() > p:
        return x
    # Convert the image to a PyTorch tensor
    img_tensor = torch.tensor(np.array(x), dtype=torch.float32) / 255.0

    # Generate Gaussian noise with the specified mean and standard deviation
    std_num = np.random.uniform(*std)
    mean_num = np.random.uniform(*mean)
    noise = torch.randn_like(img_tensor) * std_num + mean_num

    # Add the noise to the image tensor and clip the values to [0, 1]
    img_tensor = (img_tensor + noise).clamp(0.0, 1.0)

    # Convert the tensor back to a PIL image
    img_array = (img_tensor * 255.0).to(torch.uint8).numpy()
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
    MAX_LR: float = 0.000012
    MOMENTUM: float = 0.99
    WEIGHT_DECAY: float = 0.005
    GRAD_CLIP: float = 0.005
    NESTEROV: bool = True
    NUM_CLASSES: int = 2
    EPOCHS: int = 569
    BATCH_SIZE: int = 16
    DATA_DIR: str = './data'
    CHECKPOINTS_PATH: Callable = lambda epoch, m_name: f'checkpoints/img_model-{m_name}-{epoch:03d}.pkl'
    NORM_VALS: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = calc_normvals(ImageDataset(DATA_DIR, train=True))
    TRANSFORM_TEST: transforms.Compose = transforms.Compose([
                        transforms.ToTensor(),
                        #transforms.Normalize(*NORM_VALS, inplace=True)
                    ])
    TRANSFORM_TRAIN: transforms.Compose = transforms.Compose([
                        transforms.RandomCrop(80, padding=8, padding_mode='constant'),
                        transforms.GaussianBlur(3, sigma=(0.01, 1)),
                        transforms.RandomRotation(degrees=(0, 8)),
                        transforms.RandomGrayscale(),
                        transforms.RandomAutocontrast(),
                        transforms.RandomAdjustSharpness(sharpness_factor=2),
                        transforms.RandomPosterize(4),
                        transforms.RandomEqualize(),
                        transforms.Lambda(lambda x: cutout(x, 17, 17)),
                        transforms.RandomHorizontalFlip(),
                        transforms.Lambda(lambda x: gaus_noise(x)),
                        transforms.ToTensor(),
                        #transforms.Normalize(*NORM_VALS, inplace=True)
                    ])