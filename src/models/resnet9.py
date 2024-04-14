from torch import nn

class ResNet9(nn.Module):
    """
    ResNet9 model modified for 80x80 images.
    Architecture is described in blog post: https://myrtle.ai/learn/how-to-train-your-resnet/
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = ResNet9.conv_block(in_channels, 64)
        self.conv2 = ResNet9.conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(ResNet9.conv_block(128, 128), ResNet9.conv_block(128, 128))
        
        self.conv3 = ResNet9.conv_block(128, 256, pool=True)
        self.conv4 = ResNet9.conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(ResNet9.conv_block(512, 512), ResNet9.conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), 
                                        nn.Flatten(), 
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
    
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
    

    @staticmethod
    def conv_block(in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(inplace=True)]
        if pool: 
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)