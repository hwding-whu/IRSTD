import torch
import torch.nn as nn
from torchinfo import summary

class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=3, num_filters=64, num_layers=4):
        super(PatchDiscriminator, self).__init__()

        layers = []
        # 输入图像经过一系列卷积层
        layers.append(nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(num_filters * 2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            num_filters *= 2

        # 输出最后的判别结果
        layers.append(nn.Conv2d(num_filters, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    dis = PatchDiscriminator(input_channels=3, num_filters=64, num_layers=4)
    x = torch.randn(size = (1,3,128,128))
    print(dis(x).shape)
    summary(dis,input_size=(1,3,128,128))