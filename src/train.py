import torch
from torch.utils.data import DataLoader
from src.datasets import AudioDataset
from src.model import AudioCNN

def train_model(input_dir, target_dir, save_path, epochs=5, batch_size=4, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AudioDataset(input_dir, target_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AudioCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model uložen do {save_path}")