import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=7)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=15, stride=1, padding=7)
        self.conv4 = nn.Conv1d(16, 1, kernel_size=15, stride=1, padding=7)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)   # výstup má být "efektovaný"
        return x