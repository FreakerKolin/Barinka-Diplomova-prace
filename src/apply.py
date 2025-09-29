import torch
import torchaudio
from src.model import AudioCNN

def apply_model(model_path, input_wav, output_wav):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # načtení wav uvnitř funkce
    wav, sr = torchaudio.load(input_wav, normalize=True)

    # převod na mono, pokud je stereo
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # přidání batch dimenze
    wav = wav.unsqueeze(0).to(device)  # [1, 1, samples]

    with torch.no_grad():
        output = model(wav)

    # odstranění batch dimenze před uložením
    output = output.squeeze(0)

    # uložení výsledku
    torchaudio.save(output_wav, output.cpu(), sr)
    print(f"Výsledek uložen do {output_wav}")