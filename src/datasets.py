import os
import torch
import torchaudio
from torch.utils.data import Dataset

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
        inp, _ = torchaudio.load(os.path.join(self.input_dir, self.input_files[idx]))
        tgt, _ = torchaudio.load(os.path.join(self.target_dir, self.target_files[idx]))

        # zkrátíme/zarovnáme
        min_len = min(inp.shape[1], tgt.shape[1], self.segment_length)
        inp = inp[:, :min_len]
        tgt = tgt[:, :min_len]

        return inp, tgt