import torch
import torch.nn as nn


class VDSR(nn.Module):
    def __init__(self,num_channels, base_channels, num_residual):
        super(VDSR, self).__init__()

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels=num_channels,out_channels=base_channels,kernel_size=3,bias = False),
            nn.ReLU(inplace = True),
        )

        self.residual_block = []
        for _ in range(num_residual):
            self.residual_block.append(
                nn.Sequential(
                    nn.Conv2d(in_channels = base_channels,out_channels=base_channels,kernel_size = 3,bias = False),
                    nn.ReLU(inplace = True),
                )
            )

        self.residual_b = torch.nn.Sequential(*self.residual_block)

        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels=base_channels,out_channels=num_channels,kernel_size=3,bias = False),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        residual = x
        out = self.input_conv(x)
        out = self.residual_b(out)
        out = self.output_conv(out)
        out = torch.add(residual, out)

        return out

if __name__ == '__main__':
    vdsr = VDSR(num_channels=3,base_channels=3,num_residual=20) # 20 layers?
    print(vdsr)