import torch
import torch.nn as nn

class L3CNNV4(nn.Module):
    def __init__(self, in_channels=1, down_sample_factor=2, input_size=(640, 480)):
        super(L3CNNV4, self).__init__()
        self.fc_features = int(4 * (input_size[0] / down_sample_factor / 8) * (input_size[1] / down_sample_factor / 8))

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(4)

        self.fc1 = nn.Linear(self.fc_features, 128)
        self.fc2 = nn.Linear(128, 1)

        self.downsample = nn.AvgPool2d(kernel_size=down_sample_factor, stride=down_sample_factor)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        self.activation = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.downsample(x)
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = self.pool(self.activation(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x