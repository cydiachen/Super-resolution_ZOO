import torch
import torch.nn as nn


class ESPCN(nn.Module):
    def __init__(self,num_channels = 3,feature = 64, upscale_factor = 4):
        super(ESPCN, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=num_channels,out_channels=feature,kernel_size= 5,stride = 1, padding = 2)
        self.conv2 = nn.Conv2d(in_channels=feature, out_channels=feature, kernel_size=3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels=feature, out_channels=feature // 2, kernel_size=3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(in_channels=feature // 2, out_channels= 3*(upscale_factor ** 2), kernel_size= 3, stride = 1, padding = 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pixel_shuffle(out)

        return out

