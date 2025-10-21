import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Embedding, Linear, ModuleList, ReLU
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable
from transformer import TransformerEncoderLayerNoBias
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, readout=False):
        super().__init__()
        self.layers = ModuleList()
        input_fc = nn.Linear(input_dim, hidden_dim, bias=False)
        output_fc = nn.Linear(hidden_dim, output_dim, bias=False)
        self.layers.append(input_fc)
        if num_layers > 2:
            fc = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.layers.append(fc)
        self.layers.append(output_fc)
        self.num_layers = num_layers
        self.readout = readout

    def forward(self, x):
        for i in range(self.num_layers-1):
            layer = self.layers[i]
            x = layer(x)
            x = F.relu(x)
        layer = self.layers[-1]
        x = layer(x)
        return x

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, nhead=2, readout=False):
        super().__init__()
        self.layers = ModuleList()
        encoder_layer = TransformerEncoderLayerNoBias(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim, bias=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(hidden_dim, output_dim, bias=False)
        self.readout = readout
    def forward(self, x):
        x = self.transformer_encoder(x)
        #x = F.relu(x) #TODO: check if there exists one relu actiation after the attention blocks
        x = self.output_fc(x)
        return x  
    
    
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, readout=False):
        super().__init__()
        self.layers = ModuleList()
        rnn = torch.nn.GRU(input_dim,hidden_dim,num_layers,bias=False,batch_first=True)
        output_fc = nn.Linear(hidden_dim, output_dim, bias=False)
        self.layers.append(rnn)
        self.layers.append(output_fc)
        self.num_modules = 2
    def forward(self, x):
        for i in range(self.num_modules-1):
            layer = self.layers[i]
            x = layer(x)
            if type(x) is tuple:
                x=x[0]
            x = F.relu(x)
        layer = self.layers[-1]
        x = layer(x)
        return x  

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, readout=False):
        super().__init__()
        self.num_layers = num_layers
        self.layers = ModuleList()
        kernel_size=3
        
        conv_layer = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, \
                    kernel_size=kernel_size, padding=(kernel_size - 1) // 2 ,bias=False)
        self.layers.append(conv_layer)
        if num_layers > 2:
            for i in range(num_layers-1):
                conv_layer = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, \
                        kernel_size=kernel_size, padding=(kernel_size - 1) // 2 ,bias=False)
                self.layers.append(conv_layer)
        output_fc = nn.Linear(hidden_dim, output_dim, bias=False)

        self.layers.append(output_fc)
    def forward(self, x):
        x=x.permute(0,2,1) #need N,C,L, was N L C
        for i in range(self.num_layers):
            layer = self.layers[i]
            x = layer(x)
            if type(x) is tuple:
                x=x[0]
            x = F.relu(x)
        layer = self.layers[-1]
        x=x.permute(0,2,1)
        x = layer(x)
        return x 

class ContrastiveModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 encoder_flag: str = 'transformer',
                 padding_idx: int = 0,
                ):
        super().__init__()  
        self.type_embedding = Embedding(5, input_dim, padding_idx=padding_idx)  
        self.traj_embedding = Linear(2, input_dim, bias=False)

        if encoder_flag == 'mlp':
            self.traj_encoder = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, readout=True)
            self.type_encoder = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, readout=True)
        elif encoder_flag == 'cnn':
            self.traj_encoder = CNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, readout=True)
            self.type_encoder = CNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, readout=True)
        elif encoder_flag == 'rnn':
            self.traj_encoder = RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, readout=True)
            self.type_encoder = RNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, readout=True)
        elif encoder_flag == 'transformer':
            self.traj_encoder = Transformer(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, readout=True)
            self.type_encoder = Transformer(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, readout=True)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.type_embedding.reset_parameters()
        self.traj_embedding.reset_parameters()
    
    def traj_forward(self, traj):
        return self.traj_encoder(traj)
        
    def type_forward(self, types):
        return self.type_encoder(types)
        
    def forward(self, traj, types):
        traj = self.traj_embedding(traj)
        types = self.type_embedding(types)
        traj = self.traj_encoder(traj)
        types = self.type_encoder(types)
        return traj, types