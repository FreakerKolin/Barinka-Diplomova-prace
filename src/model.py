import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15, padding=7)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 16, kernel_size=15, padding=7)
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # flatten pro fc
        x = x.permute(0, 2, 1)  # [batch, seq_len, channels]
        x = self.fc(x)
        return x.permute(0, 2, 1)