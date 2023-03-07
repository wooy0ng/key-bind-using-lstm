import torch
import torch.nn as nn
import torch.nn.functional as F

class BindingModel(nn.Module):
    def __init__(self):
        super(BindingModel, self).__init__()
    
    def gaussian_normalization(self, x: torch.Tensor) -> torch.Tensor:
        '''
        gaussian normalization
        '''
        mean, std = x.mean(dim=-1).unsqueeze(-1), x.std(dim=-1).unsqueeze(-1)
        return (x - mean) / std
        
    def forward(self, x):
        return 


class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
    
    def forward(self, x):
        return


class CryptoModel(nn.Module):
    def __init__(self):
        super(CryptoModel, self).__init__()
        
        self.lstm_model = LSTMModel()
        self.binding_model = BindingModel()
        
    def forward(self, x):
        return
    
    def trainer(self, cfg):
        
        
        return
    
        