import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activation=nn.SiLU()):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        return x

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, activation=nn.SiLU()):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.bn(self.conv(x)))
        return x
    
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, activation=nn.SiLU()):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch, activation=activation)
        self.conv2 = DSConv(out_ch, out_ch, 3, padding=1, activation=activation)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class ResConv(nn.Module):
    def __init__(self, in_ch, out_ch, activation=nn.SiLU()):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch, activation=activation)
        self.conv2 = DSConv(out_ch, out_ch, 3, padding=1, activation=nn.Identity())
        self.conv3 = Conv(in_ch, out_ch, activation=nn.Identity())
        self.activation = activation

    def forward(self, x):
        x = self.conv2(self.conv1(x)) + self.conv3(x)
        return self.activation(x)

class UpDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, activation=nn.SiLU()):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, activation=activation)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x
    
class UpResConv(nn.Module):
    def __init__(self, in_ch, out_ch, activation=nn.SiLU()):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ResConv(in_ch, out_ch, activation=activation)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x

class SUNet2EnDoubleConv(nn.Module):
    def __init__(self, n_channels=16, activation=nn.SiLU()):
        super().__init__()
        self.event_stem = DSConv(5, n_channels, activation=activation)
        self.depth_stem = Conv(1, n_channels, activation=activation)

        # after fusion: standard UNet
        self.down1 = DoubleConv(2 * n_channels, 1 * n_channels, activation=activation)
        self.down2 = DoubleConv(1 * n_channels, 2 * n_channels, activation=activation)
        self.down3 = DoubleConv(2 * n_channels, 4 * n_channels, activation=activation)

        self.bottleneck = DoubleConv(4 * n_channels, 8 * n_channels, activation=activation)

        self.up1 = UpDoubleConv(8 * n_channels, 4 * n_channels, activation=activation)
        self.up2 = UpDoubleConv(4 * n_channels, 2 * n_channels, activation=activation)
        self.up3 = UpDoubleConv(2 * n_channels, 1 * n_channels, activation=activation)

        self.pool = nn.MaxPool2d(2, 2)
        self.head = nn.Conv2d(n_channels, 1, kernel_size=1)


    def forward(self, depth: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        # stems
        e = self.event_stem(event)
        d = self.depth_stem(depth)
        x = torch.cat([e, d], dim=1)

        # encoder
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x  = self.bottleneck(self.pool(x3))

        # decoder with skips
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.head(x)

        x = F.softplus(x)
        return x, None
    
    
class SUNet2EnResConv(nn.Module):
    def __init__(self, n_channels=16, activation=nn.SiLU()):
        super().__init__()
        self.event_stem = DSConv(5, n_channels, activation=activation)
        self.depth_stem = Conv(1, n_channels, activation=activation)

        # after fusion: standard UNet
        self.down1 = ResConv(2 * n_channels, 1 * n_channels, activation=activation)
        self.down2 = ResConv(1 * n_channels, 2 * n_channels, activation=activation)
        self.down3 = ResConv(2 * n_channels, 4 * n_channels, activation=activation)

        self.bottleneck = ResConv(4 * n_channels, 8 * n_channels, activation=activation)

        self.up1 = UpResConv(8 * n_channels, 4 * n_channels, activation=activation)
        self.up2 = UpResConv(4 * n_channels, 2 * n_channels, activation=activation)
        self.up3 = UpResConv(2 * n_channels, 1 * n_channels, activation=activation)

        self.pool = nn.MaxPool2d(2, 2)
        self.head = nn.Conv2d(n_channels, 1, kernel_size=1)

    def forward(self, depth: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        # stems
        e = self.event_stem(event)
        d = self.depth_stem(depth)
        x = torch.cat([e, d], dim=1)

        # encoder
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x  = self.bottleneck(self.pool(x3))

        # decoder with skips
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.head(x)

        x = F.softplus(x)
        return x, None
    
    
class SUNet2CResConv(nn.Module):
    def __init__(self, n_channels=16, activation=nn.SiLU()):
        super().__init__()
        # after fusion: standard UNet
        self.down1 = ResConv(6, 1 * n_channels, activation=activation)
        self.down2 = ResConv(1 * n_channels, 2 * n_channels, activation=activation)
        self.down3 = ResConv(2 * n_channels, 4 * n_channels, activation=activation)

        self.bottleneck = ResConv(4 * n_channels, 8 * n_channels, activation=activation)

        self.up1 = UpResConv(8 * n_channels, 4 * n_channels, activation=activation)
        self.up2 = UpResConv(4 * n_channels, 2 * n_channels, activation=activation)
        self.up3 = UpResConv(2 * n_channels, 1 * n_channels, activation=activation)

        self.pool = nn.MaxPool2d(2, 2)
        self.head = nn.Conv2d(n_channels, 1, kernel_size=1)

    def forward(self, depth: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        # stems
        x = torch.cat([event, depth], dim=1)

        # encoder
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x  = self.bottleneck(self.pool(x3))

        # decoder with skips
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        x = self.head(x)

        x = F.softplus(x)
        return x, None