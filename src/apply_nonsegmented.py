import torch
import torchaudio
from src.model import AudioCNN

def apply_model(model_path, input_wav, output_wav):
    """
    Aplikuje natrénovaný model na celý vstupní WAV soubor najednou. Výsledek se uloží do output_wav.
    Parametry: cesta k modelu, vstupní a výstupní WAV soubor.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # načtení modelu
    model = AudioCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # načtení zvuku
    wav, sr = torchaudio.load(input_wav, normalize=True)

    # převod na mono, pokud je stereo
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav = wav.to(device)

    # inferenční režim, vypne gradienty
    with torch.no_grad():
        # přidání batch dimenze: [1, channels, samples]
        processed = model(wav.unsqueeze(0)).squeeze(0)

    # přesun na CPU a uložení
    torchaudio.save(output_wav, processed.cpu(), sr)
    print(f"Výsledek uložen do {output_wav}")