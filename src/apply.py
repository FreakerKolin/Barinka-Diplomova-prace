import torch
from src.model import AudioCNN
import torchcodec.decoders as decoders
import torchcodec.encoders as encoders

def apply_model(model_path, input_wav, output_wav):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)) #načte natrénované váhy
    model.eval() # přepne model do evaluačního módu (deaktivuje dropout, batchnorm apod.)

    # načtení zvuku přes TorchCodec
    decoder = decoders.AudioDecoder()
    wav, sr = decoder.decode_file(input_wav)

    # převod na mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav = wav.unsqueeze(0).to(device)  # batch dim [1, channels, samples]

    with torch.no_grad():# vypne gradienty, nepotřebujeme je při inferenci pro snížení paměťových nároků
        output = model(wav)

    output = output.squeeze(0)  # odstranění batch dim

    # uložení výsledku přes TorchCodec
    encoder = encoders.AudioEncoder()
    encoder.encode_file(output_wav, output.cpu(), sample_rate=sr)

    print(f"Výsledek uložen do {output_wav}")