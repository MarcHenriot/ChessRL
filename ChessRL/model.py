import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_out_size(in_size, out_c, kernel_size=1, stride=1):
    in_size = torch.tensor(in_size)
    out_size = torch.div(in_size - (kernel_size - 1) - 1, stride, rounding_mode='trunc') + 1
    out_size[0] = out_c
    return out_size

class CNN(nn.Module):
    def __init__(self, observation_shape, action_size):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(observation_shape[0], 16, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)
            
        self.fc = nn.Linear(64 * 8 * 8, action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.flatten(1)
        return self.fc(x)