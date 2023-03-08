import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from transformers import WhisperModel
from torch.utils.data import DataLoader
from data.SoundDataset import SoundDataset
from data.utils import augment_sound_data


class BindingModel(nn.Module):
    def __init__(self):
        super(BindingModel, self).__init__()
        self.in_features = 384
        self.out_features = 128
        
        self.fc1 = nn.Linear(self.in_features, self.in_features*4)
        self.fc2 = nn.Linear(self.in_features*4, self.out_features)
        
        self.prelu_w = nn.Parameter(torch.Tensor(np.random.normal(0, 1, (1,))), requires_grad=True)
        
    def gaussian_normalization(self, x: torch.Tensor) -> torch.Tensor:
        ''' gaussian normalization '''
        mean, std = x.mean(dim=-1).unsqueeze(-1), x.std(dim=-1).unsqueeze(-1)
        return (x - mean) / std
        
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        # Layer1
        outputs = F.layer_norm(input_features, normalized_shape=(input_features.size(-1), ))
        outputs = F.prelu(self.fc1(outputs), weight=self.prelu_w)
        outputs = F.dropout(outputs, p=0.5)
        
        # Layer2
        outputs = torch.sigmoid(self.fc2(outputs))
        return outputs
    
    @property
    def device(self):
        return self.fc1.weight.device

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.input_size = 384
        self.hidden_size = 384      # size of embedding vector
        self.num_layers = 1
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=384,
            hidden_size=384,
            num_layers=1,
            batch_first=True
        )
        
    def forward(self, input_features: torch.Tensor, return_last_hidden_state=True) -> torch.Tensor:
        '''
            return_last_hidden_state가 True이면, (b, h) 크기의 vector를 반환합니다.
            return_last_hidden_state가 False이면, (b, s, h) 크기의 vector를 반환합니다.
        '''
        h0 = torch.zeros(self.num_layers, input_features.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, input_features.size(0), self.hidden_size).to(self.device)
        
        outputs, _ = self.lstm(input_features, (h0, c0))
        if return_last_hidden_state:
            outputs = outputs[:, -1, :]
        return outputs
    
    @property
    def device(self) -> torch.device:
        return self.lstm.weight_hh_l0.device


class CryptoModel(nn.Module):
    def __init__(self):
        super(CryptoModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Feature Extraction model
        self.whisper_encoder = WhisperModel.from_pretrained('openai/whisper-tiny').get_encoder()
        self.whisper_encoder = self.whisper_encoder.to(self.device)
        self.whisper_encoder._freeze_parameters()
                
        # LSTM model
        self.lstm_model = LSTMModel().to(self.device)
        
        # Binding model
        self.binding_model = BindingModel().to(self.device)
    
    def feature_extraction(self, input_features: torch.Tensor) -> torch.Tensor:
        input_embeds = F.gelu(self.whisper_encoder.conv1(input_features))
        input_embeds = F.gelu(self.whisper_encoder.conv2(input_embeds))
        
        input_embeds = input_embeds.permute(0, 2, 1)    # (b, 1500, 384)
        embed_pos = self.whisper_encoder.embed_positions.weight
        
        hidden_states = input_embeds + embed_pos        # positional embedding
        return hidden_states
    
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        input_features = self.feature_extraction(input_features)    # (b, 1500, 384)
        outputs = self.lstm_model(input_features)   # (b, 384)  
        outputs = self.binding_model(outputs)       # (b, key_size)
        
        return outputs
    
    def trainer(self, cfg):
        dataset = augment_sound_data(cfg.data_path)
        dataset = SoundDataset(dataset)
        data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        
        # optimizer
        optimizer = torch.optim.Adam(
            itertools.chain(self.lstm_model.parameters(), self.binding_model.parameters()),
            lr=cfg.lr
        )
        
        # criterion
        criterion = nn.MSELoss()
        
        for epoch in range(cfg.epoch):
            for idx, (path, inputs) in enumerate(data_loader):
                inputs = inputs.to(self.device)
                outputs = self(inputs)  
                
        return
    
        