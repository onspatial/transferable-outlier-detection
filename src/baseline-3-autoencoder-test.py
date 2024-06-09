#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
import torch.nn.functional as F
from data import TrajDailyDataset, TrajWeeklyDataset
from autoencoder_model import MLP, Transformer, CNN, TrajAutoencoderModel
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score


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

def test(df, groundtruth, save_dir):
    res = pd.merge(
        df, groundtruth, 
        how='left', 
        left_on=['UserId'], right_on=['agentId']
        )
    res['flag'] = (~res.type.isna()).astype(int)
    ap_score = average_precision_score(res.flag, res.AllScore)
    auc_score = roc_auc_score(res.flag, res.AllScore)
    print('All AP score:  {:.4f}, AUC score: {:.4f}'.format(ap_score, auc_score))
    type_ap_score = average_precision_score(res.flag, res.TypeScore)
    type_auc_score = roc_auc_score(res.flag, res.TypeScore)
    print('Type AP score:  {:.4f}, AUC score: {:.4f}'.format(type_ap_score, type_auc_score))
    traj_ap_score = average_precision_score(res.flag, res.TrajScore)
    traj_auc_score = roc_auc_score(res.flag, res.TrajScore)
    print('Traj AP score:  {:.4f}, AUC score: {:.4f}'.format(traj_ap_score, traj_auc_score))
    

    for topk in [5, 10, 25, 50, 100]:
        all_threshold = df.AllScore.nlargest(topk).min()
        traj_threshold, type_threshold = df.TrajScore.nlargest(topk).min(), df.TypeScore.nlargest(topk).min()
        topkall = df.loc[df.AllScore >= all_threshold]
        topktraj = df.loc[df.TrajScore >= traj_threshold]
        topktype = df.loc[df.TypeScore >= type_threshold]
        all_merged = pd.merge(topkall, groundtruth, left_on=['UserId'],  right_on=['agentId'])
        traj_merged = pd.merge(topktraj, groundtruth, left_on=['UserId'],  right_on=['agentId'])
        type_merged = pd.merge(topktype, groundtruth, left_on=['UserId'],  right_on=['agentId'])
        topk_all_hits = all_merged.shape[0]
        topk_traj_hits, topk_type_hits = traj_merged.shape[0], type_merged.shape[0]
        print(f"Top {topk} hits: All {topk_all_hits} ;Top {topk} hits: Traj {topk_traj_hits} ; Type {topk_type_hits}")


    merged = pd.merge(
        df, groundtruth, 
        how='inner', 
        left_on=['UserId'], right_on=['agentId']
        )
    fig = plt.figure()
    plt.scatter(x=df['UserId'], y=df['AllScore'], s=2)
    plt.scatter(x=merged['UserId'], y=merged['AllScore'], s=10, color='red')
    plt.title("All anomaly score")
    fig.savefig(os.path.join(save_dir,'all.jpg'), dpi=300)

    fig = plt.figure()
    plt.scatter(x=df['UserId'], y=df['TrajScore'], s=2)
    plt.scatter(x=merged['UserId'], y=merged['TrajScore'], s=10, color='red')
    plt.title("Traj anomaly score")
    fig.savefig(os.path.join(save_dir,'traj.jpg'), dpi=300)


    fig = plt.figure()
    plt.scatter(x=df['UserId'], y=df['TypeScore'], s=2)
    plt.scatter(x=merged['UserId'], y=merged['TypeScore'], s=10, color='red')
    plt.title("Type anomaly score")
    fig.savefig(os.path.join(save_dir,'type.jpg'), dpi=300)




def inference(args):
    max_weekly_seq = args.max_weekly_seq
    data_path = args.data_path
    save_path = args.save_path
    test_dataset = TrajWeeklyDataset(data_path=data_path, train=False, max_weekly_seq=max_weekly_seq)

    batch_size = args.batch_size
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

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
        raise Exception(f'{save_dir} does not exist.')
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
    model_name = 'model.pt'
    model.load_state_dict(torch.load(os.path.join(save_dir, model_name)))

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    model.eval()
    ids_list, traj_score_list, type_score_list = [], [], []
    for item in test_loader:
        ids, type_tensor, traj_tensor, attention_mask = item
        type_tensor, traj_tensor = type_tensor.to(device), traj_tensor.to(device)
        traj_score, type_score = model.score(traj_tensor, type_tensor)
        #print(traj_score.shape)
        traj_score, type_score = traj_score.mean(dim=-1), type_score.mean(dim=-1)

        traj_score_list.append(traj_score.detach().cpu().numpy())
        type_score_list.append(type_score.detach().cpu().numpy())
        ids_list.append(ids.detach().cpu().numpy())


    traj_score = np.concatenate(traj_score_list)
    type_score = np.concatenate(type_score_list)
    traj_score /= (traj_score.std()+1e-12)
    type_score /= (type_score.std()+1e-12)
    ids = np.concatenate(ids_list)
    df = pd.DataFrame(columns=['UserId','TrajScore', 'TypeScore', 'AllScore'])
    df.UserId = ids
    df.TrajScore = traj_score
    df.TypeScore = type_score
    df.AllScore = (df.TrajScore + df.TypeScore) / 2
    #print(df)
    res_name = 'anomaly_scores.csv'
    df.to_csv(os.path.join(save_dir, res_name),index=False)
    groundtruth = pd.read_csv(os.path.join(data_path, 'groundtruth.csv'))
    test(df, groundtruth, save_dir)



def run():
    args = get_args()
    inference(args)

if __name__ == "__main__":
    run()
    
