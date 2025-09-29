import torch
import torchaudio
from model import AudioCNN

def apply_model(model_path, input_wav, output_wav):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    wav, sr = torchaudio.load(input_wav)
    wav = wav.to(device)

    with torch.no_grad():
        output = model(wav)

    torchaudio.save(output_wav, output.cpu(), sr)
    print(f"Výsledek uložen do {output_wav}")