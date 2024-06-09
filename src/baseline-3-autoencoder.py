#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import argparse
import time
import logging
import json
import torch
import torch.nn.functional as F
from data import TrajDailyDataset, TrajWeeklyDataset
from autoencoder_model import MLP, Transformer, CNN, TrajAutoencoderModel
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--save_path', type=str, default='./baseline_models')
    parser.add_argument('--method', type=str, default='antoencoder')
    parser.add_argument('--encoder_type', type=str, default='mlp')
    parser.add_argument('--if_normalize', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--max_weekly_seq', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--input_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay_step_size', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=0.9)
    args = parser.parse_args()
    return args

def train(args):
    max_weekly_seq = args.max_weekly_seq
    data_path = args.data_path
    save_path = args.save_path
    train_dataset = TrajWeeklyDataset(data_path=data_path, train=True, max_weekly_seq=max_weekly_seq)

    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    input_dim, hidden_dim, output_dim = args.input_dim, args.hidden_dim, args.output_dim
    num_layers = args.num_layers
    encoder_flag = args.encoder_type
    if_normalize = args.if_normalize
    num_epochs = args.num_epochs
    dataset_name = args.data_path.split('/')[-1]

    # Convert args to dictionary
    args_dict = vars(args)
    dataset_name = args.data_path.split('/')[-1]
    save_dir = os.path.join(args.save_path, dataset_name, args.method, args.encoder_type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Save the dictionary to a JSON file
    with open(os.path.join(save_dir,'model_args.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    if encoder_flag == 'mlp':
        traj_encoder = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, readout=True)
        type_encoder = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, readout=True)
        traj_decoder = MLP(input_dim=output_dim, hidden_dim=hidden_dim, output_dim=input_dim, num_layers=num_layers, readout=False)
        type_decoder = MLP(input_dim=output_dim, hidden_dim=hidden_dim, output_dim=input_dim, num_layers=num_layers, readout=False)
    elif encoder_flag == 'transformer':
        traj_encoder = Transformer(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, readout=True)
        type_encoder = Transformer(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, readout=True)
        traj_decoder = Transformer(input_dim=output_dim, hidden_dim=hidden_dim, output_dim=input_dim, num_layers=num_layers, readout=False)
        type_decoder = Transformer(input_dim=output_dim, hidden_dim=hidden_dim, output_dim=input_dim, num_layers=num_layers, readout=False)
    elif encoder_flag == 'cnn':
        traj_encoder = CNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, readout=True)
        type_encoder = CNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, readout=True)
        traj_decoder = CNN(input_dim=output_dim, hidden_dim=hidden_dim, output_dim=input_dim, num_layers=num_layers, readout=False)
        type_decoder = CNN(input_dim=output_dim, hidden_dim=hidden_dim, output_dim=input_dim, num_layers=num_layers, readout=False)
    model = TrajAutoencoderModel(
        traj_encoder = traj_encoder,
        type_encoder = type_encoder,
        traj_decoder = traj_decoder,
        type_decoder = type_decoder,
        input_dim = input_dim,
        output_dim = output_dim,
        max_weekly_seq = max_weekly_seq
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.weight_decay_step_size, gamma=args.weight_decay)
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    log_dir = os.path.join(args.save_path, dataset_name, args.method, args.encoder_type, 'training.log')
    logging.basicConfig(filename=log_dir, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Start Training.")
    model.train()
    for epoch in range(num_epochs):
        traj_loss_list, type_loss_list = [], []
        time1 = time.perf_counter()
        for item in train_loader:
            ids, type_tensor, traj_tensor, attention_mask = item
            type_tensor, traj_tensor = type_tensor.to(device), traj_tensor.to(device)
            attention_mask = attention_mask.to(device)
            traj_loss, type_loss = model(traj_tensor, type_tensor, attention_mask)
            loss = traj_loss + type_loss 
            traj_loss_list.append(float(traj_loss))
            type_loss_list.append(float(type_loss))
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

        execution_time = time.perf_counter() - time1
        logging.info("Epoch:{:03d}, traj_loss={:.4f}, type_loss={:.4f}, execution_time={:.4f}".format(
            epoch, np.mean(traj_loss_list), np.mean(type_loss_list), execution_time
        ))
        print("Epoch:{:03d}, traj_loss={:.4f}, type_loss={:.4f},  execution_time={:.4f}".format(
            epoch, np.mean(traj_loss_list), np.mean(type_loss_list), execution_time
        ))
        scheduler.step()

    model_name = 'model.pt'

    torch.save(model.state_dict(), os.path.join(save_dir, model_name))


def run():
    args = get_args()
    train(args)

if __name__ == "__main__":
    run()
    
