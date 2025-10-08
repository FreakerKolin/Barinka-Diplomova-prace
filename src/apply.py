import torch
import torchaudio
from src.model import AudioCNN

def apply_model(model_path, input_wav, output_wav, segment_length=1024, overlap=256):
    """
    Aplikuje model na vstupní WAV soubor po segmentech s překrytím. Výsledek se uloží do output_wav.
    Parametry: cesta k modelu, vstupní a výstupní WAV soubor, délka segmentu a překrytí v počtu vzorků.
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

    # připraví buffer pro výstup
    processed = torch.zeros_like(wav)

    # okno pro překrytí (lineární interpolace)
    window = torch.linspace(0, 1, overlap, device=device)

    with torch.no_grad():  # vypne gradienty
        # Smyčka přes segmenty s překrytím
        for start in range(0, num_samples, segment_length - overlap): #zajistí překrytí
            end = min(start + segment_length, num_samples)
            segment = wav[:, start:end].unsqueeze(0)  # aktuální výřez zvuku [1, channels, samples]

            output_segment = model(segment).squeeze(0)  # [channels, samples]

            seg_len = end - start # skutečná délka segmentu (může být kratší na konci)
            if start == 0:
                # první segment (zapiš do bufferu bez okna)
                processed[:, :seg_len] = output_segment
            else:
                actual_overlap = min(overlap, seg_len) #zajistí, že poslední segment nebude mít větší překrytí než je jeho délka
                processed[:, start:start+actual_overlap] = (
                    processed[:, start:start+actual_overlap] * (1 - window[:actual_overlap]) +
                    output_segment[:, :actual_overlap] * window[:actual_overlap]
                )
                # zbytek segmentu (zapiš bez okna)
                processed[:, start+actual_overlap:start+seg_len] = output_segment[:, actual_overlap:seg_len]

    # uložení výsledku
    processed = processed.cpu()
    torchaudio.save(output_wav, processed, sr)
    print(f"Výsledek uložen do {output_wav}")