import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import os

from transformers import WhisperModel
from torch.utils.data import DataLoader
from data.SoundDataset import SoundDataset
from data.utils import augment_sound_data, load_sound_data
from utils import make_random_key, bit_to_string

# debugging
import pickle as pkl


class BindingModel(nn.Module):
    def __init__(self):
        super(BindingModel, self).__init__()
        self.in_features = 384
        self.out_features = 128
        
        self.fc1 = nn.Linear(self.in_features, self.in_features*4, bias=False)
        self.fc2 = nn.Linear(self.in_features*4, self.out_features, bias=False)
        
        self.prelu_w = nn.Parameter(torch.Tensor(np.random.normal(0, 1, (1,))), requires_grad=True)
        
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        # Layer1
        # outputs = F.layer_norm(input_features, normalized_shape=(input_features.size(-1), ))
        outputs = F.prelu(self.fc1(input_features), weight=self.prelu_w)
        outputs = F.dropout(outputs, p=0.2)
        
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
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bias=False
        )
        
    def gaussian_normalization(self, x: torch.Tensor) -> torch.Tensor:
        ''' gaussian normalization '''
        mean, std = x.mean(dim=-1).unsqueeze(-1), x.std(dim=-1).unsqueeze(-1)
        return (x - mean) / std

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, input_features.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, input_features.size(0), self.hidden_size).to(self.device)
        
        outputs, _ = self.lstm(input_features, (h0, c0))
        outputs = outputs[:, -1, :]     # get last hidden state
        # outputs = self.gaussian_normalization(outputs)
        return outputs
    
    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False
    
    @property
    def device(self) -> torch.device:
        return self.lstm.weight_hh_l0.device


class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.fc = nn.Linear(384, 1)
        
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        logit = self.fc(input_features)
        return logit
    
    @property
    def device(self) -> torch.device:
        return self.fc.weight.device
        

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
        
        # Classification model
        self.classification_model = ClassificationModel().to(self.device)
        
        # Binding model
        self.binding_model = BindingModel().to(self.device)
    
    def feature_extraction(self, input_features: torch.Tensor) -> torch.Tensor:
        input_embeds = F.gelu(self.whisper_encoder.conv1(input_features))
        input_embeds = F.gelu(self.whisper_encoder.conv2(input_embeds))
        
        input_embeds = input_embeds.permute(0, 2, 1)    # (b, 1500, 384)
        embed_pos = self.whisper_encoder.embed_positions.weight
        
        hidden_states = input_embeds + embed_pos        # positional embedding
        return hidden_states
    
    def forward(self, input_features: torch.Tensor, is_classification=False) -> torch.Tensor:
        input_features = self.feature_extraction(input_features)    # (b, 1500, 384)
        outputs = self.lstm_model(input_features)   # (b, 384)  
        
        # 의미없는 random noize 추가
        if self.training:
            fake = torch.Tensor(np.random.normal(0, 1, outputs.size())).to(self.device)
            outputs = torch.cat([outputs, fake])
        
        if is_classification is True:
            outputs = self.classification_model(outputs)    # (b, 1)
        else:
            outputs = self.binding_model(outputs)           # (b, key_size)
        return outputs
    
    def trainer(self, cfg) -> None:
        # load dataset
        dataset = augment_sound_data(cfg.data_path)
        dataset = SoundDataset(dataset)
        data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        
        # classification optimizer, bind optimizer
        optimizer_C = torch.optim.Adam(
            itertools.chain(self.lstm_model.parameters(), self.classification_model.parameters()),
            lr=cfg.clf_lr
        )
        optimizer_B = torch.optim.Adam(
            self.binding_model.parameters(),
            lr=cfg.bind_lr
        )
        
        # classification criterion, bind criterion
        criterion_C = nn.BCEWithLogitsLoss()
        criterion_B = nn.MSELoss()
        
        # ---------------------------------
        # step 1: train classification task
        # ---------------------------------
        for epoch in range(cfg.clf_epoch):
            for idx, (path, inputs) in enumerate(data_loader):
                inputs = inputs.to(self.device)
                outputs = self(inputs, is_classification=True)
                
                true_label = torch.ones(size=(outputs.size(0)//2, 1)).to(self.device)
                false_label = torch.zeros(size=(outputs.size(0)//2, 1)).to(self.device)
                batch_label = torch.cat([true_label, false_label])
                
                optimizer_C.zero_grad()
                loss = criterion_C(outputs, batch_label)
                loss.backward()
                optimizer_C.step()

                print(f"epoch : {epoch+1} \t classification loss : {loss.item():.3f}")
        
        # --------------------------
        # step 2: train binding task
        # --------------------------
        self.lstm_model._freeze_parameters()
        master_key = make_random_key(key_size=128).to(self.device, dtype=torch.float32)
    
        for epoch in range(cfg.bind_epoch):
            for idx, (path, inputs) in enumerate(data_loader):
                inputs = inputs.to(self.device)
                outputs = self(inputs)
                
                true_key = master_key.repeat(outputs.size(0)//2, 1)
                fake_key = torch.randint(0, 2, size=(outputs.size(0)//2, 128)).to(self.device, dtype=torch.float32)
                compares = torch.cat([true_key, fake_key])
                
                optimizer_B.zero_grad()
                loss = criterion_B(outputs, compares)
                loss.backward()
                optimizer_B.step()
                
                print(f"epoch : {epoch+1} \t bind loss : {loss.item():.3f}")
        
        # save model
        self.save_model(cfg.model_path)
    
    @torch.no_grad()
    def test(self, cfg) -> None:
        self.load_state_dict(torch.load(cfg.model_path))
        self.eval()
        
        # load dataset
        dataset = load_sound_data(cfg.data_path, return_mel=True)
        dataset = SoundDataset(dataset)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        for idx, (path, inputs) in enumerate(data_loader):
            print(f"path : ", path)
            
            inputs = inputs.to(self.device)
            predicted = self(inputs)
            
            predicted = torch.where(predicted < 0.5, 0, 1)
            predicted = predicted.squeeze().cpu().detach()
            
            answer = make_random_key(key_size=128)
            
            print(f"predicted \t : {bit_to_string(predicted)}")
            print(f"answer \t\t : {bit_to_string(answer)}\n\n")
            
    
    def save_model(self, model_path: str) -> None:
        assert os.path.exists('models'), 'model_path error occurred'
        torch.save(self.state_dict(), model_path)