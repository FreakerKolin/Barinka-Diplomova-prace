import os
import torch
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):#načte soubory z adresářů MUSÍ SE JMENOVAT STEJNĚ a připraví na trénink
    def __init__(self, input_dir, target_dir, segment_length=512):
        self.input_files = sorted(os.listdir(input_dir))
        self.target_files = sorted(os.listdir(target_dir))
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.segment_length = segment_length

    def __len__(self):#vrátí počet souborů
        return len(self.input_files)

    def __getitem__(self, idx):#vrátí načtený a připravený pár souborů
        input_path = os.path.join(self.input_dir, self.input_files[idx])
        target_path = os.path.join(self.target_dir, self.target_files[idx])

        inp, sr_inp = torchaudio.load(input_path)
        tgt, sr_tgt = torchaudio.load(target_path)

        # převod na mono, zprůměrování kanálů
        if inp.shape[0] > 1:
            inp = inp.mean(dim=0, keepdim=True)
        if tgt.shape[0] > 1:
            tgt = tgt.mean(dim=0, keepdim=True)

       # ořízne oba signály na stejnou délku, maximálně segment_length
        min_len = min(inp.shape[1], tgt.shape[1], self.segment_length)
        inp = inp[:, :min_len]
        tgt = tgt[:, :min_len]

        return inp, tgt