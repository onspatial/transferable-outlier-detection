from pickle import TRUE
import numpy as np
import pandas as pd
from tqdm import tqdm 
import torch
import os
import argparse
import time
import torch.nn.functional as F
from data import TrajDailyDataset
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--save_path', type=str, default='./baseline_models')
    parser.add_argument('--method', type=str, default='trad')
    parser.add_argument('--if_normalize', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--max_daily_seq', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()
    return args

def encoder_embed(type_tensor, traj_tensor, mini_batch_size, args, normalize=True):
    type_tensor = type_tensor.flatten(start_dim=1, end_dim=2)
    traj_tensor = traj_tensor.flatten(start_dim=1, end_dim=2)

    traj_encoding = torch.sqrt(((traj_tensor[:,:,:-1] - traj_tensor[:,:,1:])**2).sum(dim=-1))
    type_encoding = F.one_hot(type_tensor, num_classes=5).float()
    
    traj_encoding = traj_encoding.reshape(mini_batch_size, -1, args.max_daily_seq, 1)
    traj_encoding = traj_encoding.sum(dim=-2)
    type_encoding = type_encoding.view(mini_batch_size, -1, args.max_daily_seq, 5)
    type_encoding = type_encoding.sum(dim=-2)
    if normalize:
        traj_encoding = F.normalize(traj_encoding, p=2.0, dim=-1)
        type_encoding = F.normalize(type_encoding, p=2.0, dim=-1)
    return traj_encoding, type_encoding 


def inference(args):
    data_path = args.data_path
    save_path = args.save_path

    batch_size = args.batch_size
    max_daily_seq = args.max_daily_seq
    if_normalize = args.if_normalize

    dataset_name = args.data_path.split('/')[-1]
    save_dir = os.path.join(save_path, dataset_name, args.method)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    train_dataset = TrajDailyDataset(data_path=data_path, train=True, max_daily_seq=max_daily_seq)
    test_dataset = TrajDailyDataset(data_path=data_path, train=False, max_daily_seq=max_daily_seq)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    ids_list, traj_score_list, type_score_list = [], [], []
    for item_train, item_test in tqdm(zip(train_loader, test_loader)):
        ids_train, type_tensor_train, traj_tensor_train, attention_mask_train, dayofweek_train = item_train
        ids_test, type_tensor_test, traj_tensor_test, attention_mask_test, dayofweek_test = item_test
        assert (ids_train == ids_test).all()
        mini_batch_size = ids_train.shape[0]
        type_tensor_train, traj_tensor_train, dayofweek_train = type_tensor_train.to(device), traj_tensor_train.to(device), dayofweek_train.to(device)
        type_tensor_test, traj_tensor_test, dayofweek_test = type_tensor_test.to(device), traj_tensor_test.to(device), dayofweek_test.to(device)

        # encode
        traj_encoding_train, type_encoding_train = encoder_embed(type_tensor_train, traj_tensor_train, mini_batch_size, args, normalize=if_normalize)
        traj_encoding_test, type_encoding_test = encoder_embed(type_tensor_test, traj_tensor_test, mini_batch_size, args, normalize=if_normalize)

        # score
        traj_encoding_by_days_train = torch.bmm(F.one_hot(dayofweek_train).permute((0,2,1)).float(), traj_encoding_train)
        traj_encoding_by_days_test = torch.bmm(F.one_hot(dayofweek_test).permute((0,2,1)).float(), traj_encoding_test)

        type_encoding_by_days_train = torch.bmm(F.one_hot(dayofweek_train).permute((0,2,1)).float(), type_encoding_train)
        type_encoding_by_days_test = torch.bmm(F.one_hot(dayofweek_test).permute((0,2,1)).float(), type_encoding_test)

        if TRUE:
            traj_encoding_by_days_train = F.normalize(traj_encoding_by_days_train, p=2.0, dim=-1)
            traj_encoding_by_days_test = F.normalize(traj_encoding_by_days_test, p=2.0, dim=-1)
            type_encoding_by_days_train = F.normalize(type_encoding_by_days_train, p=2.0, dim=-1)
            type_encoding_by_days_test = F.normalize(type_encoding_by_days_test, p=2.0, dim=-1)

        traj_sim_score = (traj_encoding_by_days_train * traj_encoding_by_days_test).sum(dim=-1).mean(dim=-1)
        type_sim_score = (type_encoding_by_days_train * type_encoding_by_days_test).sum(dim=-1).mean(dim=-1)

        traj_score_list.append(traj_sim_score.detach().cpu().numpy())
        type_score_list.append(type_sim_score.detach().cpu().numpy())
        ids_list.append(ids_train.detach().cpu().numpy())

    traj_score = np.concatenate(traj_score_list)
    type_score = np.concatenate(type_score_list)
    ids = np.concatenate(ids_list)
    df = pd.DataFrame(columns=['UserId','TrajScore', 'TypeScore', 'AllScore'])
    df.UserId = ids
    df.TrajScore = traj_score
    df.TypeScore = type_score
    df.AllScore = (df.TrajScore + df.TypeScore) / 2
    res_name = 'anomaly_scores.csv'
    df.to_csv(os.path.join(save_dir, res_name),index=False)

def run():
    args = get_args()
    inference(args)

if __name__ == "__main__":
    run()
