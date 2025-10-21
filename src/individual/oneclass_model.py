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
        if self.readout:
            x = x.sum(dim=-2)
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
        if self.readout:
            x = x.sum(dim=-2)
        return x  

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, readout=False):
        super().__init__()
        self.num_layers = num_layers
        self.layers = ModuleList()
        self.readout = readout
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
        n, weeks, l, e = x.shape
        x = x.flatten(start_dim=1, end_dim=2)
        x=x.permute(0,2,1) #need N,C,L, was N L C
        for i in range(self.num_layers):
            layer = self.layers[i]
            x = layer(x)
            if type(x) is tuple:
                x=x[0]
            x = F.relu(x)
        layer = self.layers[-1]
        x=x.permute(0,2,1)
        #print(x.shape)
        x = x.reshape(n, weeks, l, -1)
        x = layer(x)
        if self.readout:
            x = x.sum(dim=-2)
        return x 

class OneClassModel(nn.Module):
    def __init__(self,
                 encoder: str, 
                 output_dim: int, 
                 r: float = .1
                ):
        super().__init__()
        self.encoder = encoder
        self.center = torch.nn.Parameter(
            data=torch.ones(output_dim, dtype=torch.float), 
            requires_grad=False
        ).view(1,-1)
        self.r = torch.nn.Parameter(
            data=torch.tensor(r, dtype=torch.float), 
            requires_grad=True
        )
    def send_to_device(self, device):
        self.center = self.center.to(device)
        
    def encode(self, x):
        return self.encoder(x)
    
    def score(self, embeds):
        #embeds = self.encode(x)
        score = torch.cdist(embeds, self.center.view(1,-1))
        return score
    
    def oneclassLoss(self, embeds):
        #print("embeds.shape:",embeds.shape)
        #print("self.center.shape:",self.center.shape)
        distances = torch.cdist(embeds, self.center)
        #print("distances.shape",distances.shape)
        loss_ = torch.maximum(
                distances - self.r, 
                torch.tensor([0.], dtype=torch.float, device=distances.device)
            )
        #print("loss_.shape", loss_.shape)
        return torch.mean(
            loss_
        )
class TrajOneClassModel(nn.Module):
    def __init__(self,
                 traj_encoder: str, 
                 type_encoder: str, 
                 input_dim: int, 
                 output_dim: int, 
                 r: float = .1,
                 padding_idx: int = 0
                ):
        super().__init__()
        self.traj_oneclass = OneClassModel(traj_encoder, output_dim, r)
        self.type_oneclass = OneClassModel(type_encoder, output_dim, r)    
        self.type_embedding = Embedding(5, input_dim, padding_idx=padding_idx)  
        self.traj_embedding = Linear(2, input_dim, bias=False)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.type_embedding.reset_parameters()
        self.traj_embedding.reset_parameters()
    
    def traj_forward(self, traj):
        self.traj_oneclass.encode(traj)
        
    def type_forward(self, types):
        self.type_oneclass.encode(types)
        
    def encode(self, traj, types):
        types = self.type_embedding(types)
        traj = self.traj_embedding(traj)
        
        traj = self.traj_oneclass.encode(traj)
        types = self.type_oneclass.encode(types)
        return traj, types
    
    def loss(self, traj, types):
        traj, types = self.encode(traj, types)
        traj_loss = self.traj_oneclass.oneclassLoss(traj)
        type_loss = self.type_oneclass.oneclassLoss(types)
        return traj_loss, type_loss
    
    def score(self, traj, types):
        traj, types = self.encode(traj, types)
        traj_score = self.traj_oneclass.score(traj)
        type_score = self.type_oneclass.score(types)
        return traj_score, type_score