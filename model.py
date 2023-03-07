import torch
import torch.nn as nn
import torch.optim as optim

'''
    [model.py]
    LSTM + Classification model / Key generation model 별로 구성된 class 제공
    (mode에서 각각의 class 호출해서 사용)
'''


# LSTM + Classification model
class LSTMClassification(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, sequence_length, num_layers , device):
        super(LSTMClassification, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.device = device
        
        # clf model - network 설정 (선언)
        self.lstm = nn.LSTM(
            input_size=self.in_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.clf = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.out_size),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=5e-3)
        self.criterion = nn.MSELoss()
        self.iteration = 0
        self.losses = []
    
    # clf-model network 실행
    def forward(self, x):
        out, (h, c) = self.lstm(x)
        self.context = out[:, -1, :]
        result = self.clf(self.context)
        return result

    def train(self, inputs, labels):
        inputs = inputs.to(self.device)
        labels = labels.view(-1, 1).to(self.device)
        predicted = self.forward(inputs)
        
        loss = self.criterion(predicted, labels)
        self.losses.append(loss.item())
        self.iteration += 1
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Key-generation model
class KeyGeneration(nn.Module):
    def __init__(self, args):
        super(KeyGeneration, self).__init__()
        self.in_size = args.hidden_size
        self.out_size = args.master_key_size

        # key-gen model - network layer 설정 (선언)
        self.layer1 = nn.Sequential(
            nn.LayerNorm(self.in_size),
            nn.Linear(self.in_size, self.in_size * 4),
            nn.PReLU(),
            #nn.Linear(self.in_size, self.in_size),
            #nn.Sigmoid()
        )

        self.drop = nn.Dropout(p=0.5)

        self.layer2 = nn.Sequential(
            nn.Linear(self.in_size * 4, self.out_size),
            nn.Sigmoid()
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=5e-3)

        self.losses = []
        self.iteration = 0
    
    # key-gen model - network layer 실행
    def forward(self, x):
        x = x.clone()
        out = self.layer1(x)
        out = self.drop(out)
        out = self.layer2(out)
        return out
    
    def train(self, inputs, labels, key):
        inputs = inputs.to(self.device)

        labels = labels.to(self.device)
        outputs = self.forward(inputs)
        keys = key.repeat(outputs.shape[0], 1)
        
        for idx, label in enumerate(labels):
            if not label: 
                keys[idx] = torch.randint(0, 2, key.shape)

        loss = self.criterion(outputs, keys)
        self.iteration += 1
        self.losses.append(loss.item())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def test(self, inputs, labels, device):
        out = self.forward(inputs)
        return out