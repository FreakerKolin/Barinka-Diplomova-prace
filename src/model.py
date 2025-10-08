import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self):
        """
        1D konvoluční neuronová síť pro modelování a reprodukci zvukových efektů.
        Vstup: [batch, channels, samples], Výstup: [batch, channels, samples]
        """
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=10, padding='same')#1D konvoluce (vstupní kanály = 1, výstupní = 16 (nastavitelné)), kernel size = velikost filtru - nastavitelné
        self.conv2 = nn.Conv1d(16, 16, kernel_size=10, padding='same') #druhá konvoluční vrstva, příjmá 16 kanálů z první vrstvy a vrací 16 kanálů
        self.relu = nn.ReLU() #ReLU aktivační funkce, dá záporné hodnoty na nulu
        self.fc = nn.Conv1d(16, 1, kernel_size=1)  # po konvoluci sníží počet kanálů zpět na 1 (mono audio)

    def forward(self, x):
        """
        Průchod dat sítí. x je vstupní tensor tvaru [batch, channels, samples].
        """
        x = self.relu(self.conv1(x)) 
        x = self.relu(self.conv2(x))
        x = self.fc(x) 
        return x