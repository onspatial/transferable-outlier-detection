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

class AutoencoderModel(nn.Module):
    def __init__(self,
                 encoder: str, 
                 decoder: str, 
                 max_weekly_seq: int,
                 output_dim: int
                ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.unpooling = Linear(
            output_dim, max_weekly_seq*output_dim, bias=False
            )
        
    def encode(self, x):
        z = self.encoder(x)
        z = self.unpooling(z)
        return z

    def decode(self, z):
        return self.decoder(z)

class TrajAutoencoderModel(nn.Module):
    def __init__(self,
                 traj_encoder: str, 
                 traj_decoder: str, 
                 type_encoder: str, 
                 type_decoder: str, 
                 input_dim: int, 
                 output_dim: int, 
                 max_weekly_seq: int,
                ):
        super().__init__()
        self.traj_autoencoder = AutoencoderModel(traj_encoder, traj_decoder, max_weekly_seq, output_dim)
        self.type_autoencoder = AutoencoderModel(type_encoder, type_decoder, max_weekly_seq, output_dim)    
        self.type_embedding = Linear(5, input_dim, bias=False)  
        self.traj_embedding = Linear(2, input_dim, bias=False)
        self.type_recon_embedding = Linear(input_dim, 5, bias=False)  
        self.traj_recon_embedding = Linear(input_dim, 2, bias=False)
        self.type_criterion = nn.CrossEntropyLoss(reduction='none')
        self.traj_criterion = nn.MSELoss(reduction='none')
        self.max_weekly_seq = max_weekly_seq
        self.output_dim = output_dim
        self.reset_parameters()
        
    def reset_parameters(self):
        self.type_embedding.reset_parameters()
        self.traj_embedding.reset_parameters()
        self.type_recon_embedding.reset_parameters()
        self.traj_recon_embedding.reset_parameters()
        
    def reconstruct(self, traj, types):
        types_embed = self.type_embedding(types)
        traj_embed = self.traj_embedding(traj)
        batch_size, weeks, _, _ = types_embed.shape
        
        types_z = self.type_autoencoder.encode(types_embed)
        traj_z = self.traj_autoencoder.encode(traj_embed)

        types_z = types_z.reshape(batch_size, weeks, self.max_weekly_seq, self.output_dim)
        traj_z = types_z.reshape(batch_size, weeks, self.max_weekly_seq, self.output_dim)

        types_recon = self.type_autoencoder.decode(types_z)
        traj_recon = self.traj_autoencoder.decode(traj_z)

        types_recon = self.type_recon_embedding(types_recon)
        traj_recon = self.traj_recon_embedding(traj_recon)
        return traj_recon, types_recon
    
    def forward(self, traj, types, attention_mask):
        types = F.one_hot(types, num_classes=5).float()
        traj_recon, types_recon= self.reconstruct(traj, types)
        traj_loss = self.traj_criterion(traj_recon.reshape(-1,2), traj.reshape(-1,2))
        type_loss = self.type_criterion(types_recon.reshape(-1,5), types.reshape(-1,5))
        traj_loss = traj_loss.sum(dim=-1)
        attention_mask = attention_mask.reshape(-1)
        traj_loss, type_loss = traj_loss[attention_mask], type_loss[attention_mask]
        return torch.mean(traj_loss), torch.mean(type_loss)

    def score(self, traj, types):
        types = F.one_hot(types, num_classes=5).float()
        traj_recon, types_recon= self.reconstruct(traj, types)
        traj_loss = self.traj_criterion(traj_recon, traj)
        type_loss = self.type_criterion(types_recon, types)
        traj_loss = traj_loss.sum(dim=-1)
        #print(traj_loss.shape, type_loss.shape)
        return traj_loss, type_loss