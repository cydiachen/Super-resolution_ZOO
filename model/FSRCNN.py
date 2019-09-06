import torch
import torch.nn as nn


class FSRCNN(nn.Module):
    def __init__(self, num_channels, upscale_factor, feature_channels=64, mid_level=12, mappings=4):
        super(FSRCNN, self).__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=feature_channels, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
        )

        self.shrinking = nn.Sequential(
            nn.Conv2d(in_channels=feature_channels, out_channels=mid_level, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
        )

        self.Mappings = []
        for _ in range(mappings):
            self.Mappings.append(nn.Sequential(
                nn.Conv2d(in_channels=mid_level, out_channels=mid_level, kernel_size=3, stride=1, padding=1),
                nn.PReLU(),
            ))

        self.Mappings.append(nn.Sequential(
            nn.Conv2d(in_channels=mid_level, out_channels=feature_channels, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
        ))

        self.mid_part = torch.nn.Sequential(*self.Mappings)

        self.last_part = nn.ConvTranspose2d(in_channels=feature_channels, out_channels=num_channels, kernel_size=9,
                                            stride=upscale_factor, padding=3, output_padding=1)

    def forward(self, x):
        out = self.feature_extraction(x)
        out = self.shrinking(out)
        out = self.mid_part(out)
        out = self.last_part(out)

        return out

if __name__ == '__main__':
    fsrcnn = FSRCNN(num_channels=3,upscale_factor=4)
    print(fsrcnn)