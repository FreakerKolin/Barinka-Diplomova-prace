import torch
import torchaudio
from src.model import AudioCNN

def apply_model(model_path, input_wav, output_wav, segment_length=1024):
    """
    Aplikuje model na vstupní WAV soubor po segmentech a měří čas zpracování každého segmentu.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # načtení zvuku
    wav, sr = torchaudio.load(input_wav, normalize=True)

    # převod na mono, pokud je stereo
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav = wav.to(device)
    num_samples = wav.shape[1]

    processed = torch.zeros_like(wav)


    # uložení výsledku
    torchaudio.save(output_wav, processed.cpu(), sr)
    print(f"Výsledek uložen do {output_wav}")