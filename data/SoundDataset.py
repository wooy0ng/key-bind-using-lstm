import torch
import whisper
import numpy as np

from torch.utils.data import Dataset
from typing import Dict, Tuple


class SoundDataset(Dataset):
    def __init__(self, dataset: Dict):
        self.dataset = dataset
        self.dataset['audio'] = [self.to_mel_spectrogram(audio) for audio in self.dataset['audio']]
    
    def __len__(self):
        return len(self.dataset['audio'])

    def __getitem__(self, idx) -> Tuple[str, torch.Tensor]:
        return self.dataset['path'][idx], torch.Tensor(self.dataset['audio'][idx])
        
    def to_mel_spectrogram(self, audio):
        mel = whisper.log_mel_spectrogram(np.array(audio, dtype=np.float32))    # (80, 3000)
        return mel