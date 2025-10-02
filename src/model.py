import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15, padding=7)#1D konvoluce, vidí 15 vzorků najednou, padding 7 aby zachoval stejnou délku výstupu jako je vstup
        self.conv2 = nn.Conv1d(16, 16, kernel_size=15, padding=7) #druhá konvoluční vrstva
        self.relu = nn.ReLU() #ReLU aktivační funkce
        self.fc = nn.Conv1d(16, 1, kernel_size=1)  # po konvoluci sníží počet kanálů zpět na 1

    def forward(self, x):
        # x: [batch, channels, samples]
        x = self.relu(self.conv1(x)) 
        x = self.relu(self.conv2(x))
        x = self.fc(x) 
        return x