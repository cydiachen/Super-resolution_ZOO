import torch
import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self, num_channels, base_filter = 64,upscale_factor = 2):
        super(SRCNN, self).__init__()

        self.head = torch.nn.Sequential(
            nn.Conv2d(in_channels = num_channels,out_channels = base_filter,kernel_size=9,stride = 1,padding = 4,bias = True),
            nn.ReLU(inplace = True),
        )

        self.mapping = torch.nn.Sequential(
            nn.Conv2d(in_channels = base_filter,out_channels = base_filter // 2, kernel_size =1 ,bias = True),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = base_filter // 2 ,out_channels = num_channels , kernel_size=5, stride = 1,padding = 2,bias = True),

        )

    def forward(self, x):
        out = self.head(x)
        out = self.mapping(out)

        return out


if __name__ == '__main__':
    srcnn = SRCNN(num_channels=3,upscale_factor=4)
    print(srcnn)