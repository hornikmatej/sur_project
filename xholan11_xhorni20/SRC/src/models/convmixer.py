from torch import nn
from src.models.basemodel import BaseModel

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(BaseModel):
    """
    ConvMixer model.
    Architecture is described in paper: https://arxiv.org/abs/2109.04454
    dim = 1536/768, depth = 20/32
    """
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, n_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for i in range(depth)],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, n_classes)
        )

    def forward(self, x):
        return self.model(x)