import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from src.datasets import AudioDataset
from src.model import AudioCNN
import time
import os

def train_model(input_dir, target_dir, save_dir="models", epochs=100, batch_size=1, segment_length=512):
    """
    Trénuje 1D CNN model na párech vstupních a cílových WAV souborů.
    Model se uloží s timestampem do složky save_dir (nastaven na složku "models").
    Parametry: cesta ke vstupní složce, cílové složce, složce pro uložení modelu, počet epoch, velikost batch a délka segmentu pro trénink v počtu vzorků.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AudioDataset(input_dir, target_dir, segment_length=segment_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AudioCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.6f}")

    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # timestamp s časem
    model_filename = os.path.join(save_dir, f"cnn_model_{timestamp}.pth")
    torch.save(model.state_dict(), model_filename)
    print(f"Model uložen jako {model_filename}")