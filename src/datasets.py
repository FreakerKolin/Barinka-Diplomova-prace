import os
import torch
from torch.utils.data import Dataset
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, input_dir, target_dir, segment_length=16384):
        self.input_files = sorted(os.listdir(input_dir))
        self.target_files = sorted(os.listdir(target_dir))
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.segment_length = segment_length

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        waveform_in, sr_in = torchaudio.load(
            os.path.join(self.input_dir, self.input_files[idx]), normalize=True
        )
        waveform_tgt, sr_tgt = torchaudio.load(
            os.path.join(self.target_dir, self.target_files[idx]), normalize=True
        )

        # převod na mono, pokud je stereo
        if waveform_in.shape[0] > 1:
            waveform_in = waveform_in.mean(dim=0, keepdim=True)
        if waveform_tgt.shape[0] > 1:
            waveform_tgt = waveform_tgt.mean(dim=0, keepdim=True)

        # zkrácení na segment_length
        min_len = min(waveform_in.shape[1], waveform_tgt.shape[1], self.segment_length)
        waveform_in = waveform_in[:, :min_len]
        waveform_tgt = waveform_tgt[:, :min_len]

        return waveform_in, waveform_tgt