import torch.nn as nn

class BaseModel(nn.Module):
    """
    Base class for all models.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass of the model.
        """
        raise NotImplementedError

    def get_model_name(self):
        """
        Returns the name of the model.
        """
        return self.__class__.__name__

    def count_parameters(self):
        """
        Returns the number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)