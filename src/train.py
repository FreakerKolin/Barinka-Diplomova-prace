import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from src.datasets import AudioDataset
from src.model import AudioCNN

def train_model(input_dir, target_dir, save_path, epochs=100, batch_size=1, segment_length=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #použítí gpu pokud je dostupné
    dataset = AudioDataset(input_dir, target_dir, segment_length=segment_length) #vytvoření datasetu
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #a loaderu

    model = AudioCNN().to(device) #inicializace modelu
    criterion = nn.MSELoss()#definice loss funkce (zde MSE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)#definice optimizéru (zde adam a learning rate 0.001)

    for epoch in range(epochs):#trénování modelu
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

    torch.save(model.state_dict(), save_path)
    print(f"Model uložen do {save_path}")