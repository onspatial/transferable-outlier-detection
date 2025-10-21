#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import argparse
import time
import torch
import torch.nn.functional as F
from data import TrajDailyDataset
from contrastive_model import ContrastiveModel
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


def compute_flag(dayofweek):
    onehot_dayofweek = F.one_hot(dayofweek)
    onehot_dayofweek = onehot_dayofweek.float()
    inner_flag = torch.bmm(onehot_dayofweek, onehot_dayofweek.permute((0,2,1)))
    inter_flag = 1 - torch.bmm(onehot_dayofweek, onehot_dayofweek.permute((0,2,1)))
    return inner_flag, inter_flag

def ContrastiveLoss(encoding, inner_flag, inter_flag, args, criterion, if_normalize=True):
    encoding = encoding.view(-1, args.window_size, args.max_daily_seq, args.output_dim)
    encoding = encoding.sum(dim=-2)
    if if_normalize:
        encoding = F.normalize(encoding, p=2.0, dim=-1)

    # postive pairs
    inner_crossproduct = torch.bmm(encoding, encoding.permute((0,2,1)))

    # negative pairs
    inter_crossproduct = torch.bmm(encoding, torch.roll(encoding, 1, dims=0).permute((0,2,1)))

    # compute constrastive loss
    logits = inner_flag * inner_crossproduct + inter_flag * inter_crossproduct
    labels = inner_flag.float()
    loss = criterion(logits, labels)

    return loss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--save_path', type=str, default='../models')
    parser.add_argument('--encoder_type', type=str, default='transformer')
    parser.add_argument('--if_normalize', type=bool, default=True)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--max_daily_seq', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--window_size', type=int, default=28)
    parser.add_argument('--input_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay_step_size', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=0.9)
    args = parser.parse_args()
    return args

def train(args):
    max_daily_seq = args.max_daily_seq
    data_path = args.data_path
    save_path = args.save_path
    train_dataset = TrajDailyDataset(data_path=data_path, train=True, max_daily_seq=max_daily_seq, suffix=args.suffix)

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    input_dim, hidden_dim, output_dim = args.input_dim, args.hidden_dim, args.output_dim
    num_layers = args.num_layers
    encoder_flag = args.encoder_type
    if_normalize = args.if_normalize
    model = ContrastiveModel(
        input_dim = input_dim,
        hidden_dim = hidden_dim,
        output_dim = output_dim,
        num_layers = num_layers,
        encoder_flag = encoder_flag
    )

    num_epochs = args.num_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.weight_decay_step_size, gamma=args.weight_decay)
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        traj_loss_list, type_loss_list = [], []
        time1 = time.perf_counter()
        for item in train_loader:
            ids, type_tensor, traj_tensor, attention_mask, dayofweek = item
            type_tensor = type_tensor[:,46:,:]
            traj_tensor = traj_tensor[:,46:,:,:]
            dayofweek = dayofweek[:,46:]
            dayofweek = dayofweek.reshape(-1,args.window_size)
            type_tensor = type_tensor.flatten(start_dim=1, end_dim=2)
            traj_tensor = traj_tensor.flatten(start_dim=1, end_dim=2)
            type_tensor, traj_tensor, dayofweek = type_tensor.to(device), traj_tensor.to(device), dayofweek.to(device)
            #attention_mask = attention_mask.flatten(start_dim=1, end_dim=2)
            traj_encoding, type_encoding = model(traj_tensor, type_tensor)
            inner_flag, inter_flag = compute_flag(dayofweek)

            traj_loss = ContrastiveLoss(traj_encoding, inner_flag, inter_flag, args, criterion, if_normalize)
            type_loss = ContrastiveLoss(type_encoding, inner_flag, inter_flag, args, criterion, if_normalize)
            traj_loss_list.append(float(traj_loss))
            type_loss_list.append(float(type_loss))

            loss = traj_loss + type_loss
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
        execution_time = time.perf_counter() - time1
        print("Epoch:{:03d}, traj_loss={:.4f}, type_loss={:.4f}, execution_time={:.4f}".format(
            epoch, np.mean(traj_loss_list), np.mean(type_loss_list), execution_time
        ))
        scheduler.step()

    model_name = 'b'+str(batch_size)+'l'+str(num_layers)+'d'+str(hidden_dim)+'e'+str(num_epochs)+'ws'+str(args.window_size)+'_normal'+str(int(if_normalize))+'_slidingwindow.pt'

    torch.save(model.state_dict(), os.path.join(save_path, model_name))

def run():
    args = get_args()
    train(args)

if __name__ == "__main__":
    run()
    
