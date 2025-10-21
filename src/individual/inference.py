import numpy as np
import pandas as pd
from tqdm import tqdm 
import torch
import os
import argparse
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data import TrajDailyDataset
from contrastive_model import MLP, Transformer, ContrastiveModel
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--save_path', type=str, default='../models')
    parser.add_argument('--res_path', type=str, default='../res')
    parser.add_argument('--encoder_type', type=str, default='transformer')
    parser.add_argument('--if_normalize', type=bool, default=True)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--max_daily_seq', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--if_sliding_window', type=bool, default=False)
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

def encoder_embed(model, type_tensor, traj_tensor, mini_batch_size, args, normalize=True):
    type_tensor = type_tensor.flatten(start_dim=1, end_dim=2)
    traj_tensor = traj_tensor.flatten(start_dim=1, end_dim=2)
    traj_encoding, type_encoding = model(traj_tensor, type_tensor)
    traj_encoding = traj_encoding.view(mini_batch_size, -1, args.max_daily_seq, args.output_dim)
    traj_encoding = traj_encoding.sum(dim=-2)
    type_encoding = type_encoding.view(mini_batch_size, -1, args.max_daily_seq, args.output_dim)
    type_encoding = type_encoding.sum(dim=-2)
    if normalize:
        traj_encoding = F.normalize(traj_encoding, p=2.0, dim=-1)
        type_encoding = F.normalize(type_encoding, p=2.0, dim=-1)
    return traj_encoding, type_encoding 


def inference(args):
    data_path = args.data_path
    save_path = args.save_path
    res_path = args.res_path
    num_epochs = args.num_epochs

    batch_size = args.batch_size
    max_daily_seq = args.max_daily_seq

    input_dim, hidden_dim, output_dim = args.input_dim, args.hidden_dim, args.output_dim
    num_layers = args.num_layers
    encoder_flag = args.encoder_type
    if_normalize = args.if_normalize
    if_sliding_window = args.if_sliding_window
    model = ContrastiveModel(
        input_dim = input_dim,
        hidden_dim = hidden_dim,
        output_dim = output_dim,
        num_layers = num_layers,
        encoder_flag = encoder_flag
    )
    dataset_name = args.data_path.split('/')[-1]
    folder_name = 'b'+str(batch_size)+'l'+str(num_layers)+'d'+str(hidden_dim)+'e'+str(num_epochs)+'_normal'+str(int(if_normalize))
    save_dir = os.path.join(args.save_path, dataset_name, args.encoder_type, folder_name)
    
    #if if_sliding_window:
    #    print('Use sliding window model.')
    #    model_name = 'b'+str(batch_size)+'l'+str(num_layers)+'d'+str(hidden_dim)+'e'+str(num_epochs)+'ws'+str(args.window_size)+'_normal'+str(int(if_normalize))+'_slidingwindow.pt'
    #else:
    #    print('Use long sequence model.')
    #    model_name = 'b'+str(batch_size)+'l'+str(num_layers)+'d'+str(hidden_dim)+'e'+str(num_epochs)+'_normal'+str(int(if_normalize))+'.pt'
    model_name = 'model.pt'
    if not os.path.exists(os.path.join(save_dir, model_name)):
        raise Exception(f'{os.path.join(save_dir, model_name)} does not exist.')
    model.load_state_dict(torch.load(os.path.join(save_dir, model_name)))
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    train_dataset = TrajDailyDataset(data_path=data_path, train=True, max_daily_seq=max_daily_seq)
    test_dataset = TrajDailyDataset(data_path=data_path, train=False, max_daily_seq=max_daily_seq)
    num_dates_train, num_dates_test = train_dataset.num_dates(), test_dataset.num_dates()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    ids_list, traj_score_list, type_score_list = [], [], []
    for item_train, item_test in tqdm(zip(train_loader, test_loader)):
        ids_train, type_tensor_train, traj_tensor_train, attention_mask_train, dayofweek_train = item_train
        ids_test, type_tensor_test, traj_tensor_test, attention_mask_test, dayofweek_test = item_test
        #flag_mask = torch.tensor([item in ids_test for item in ids_train])
        #ids_train, type_tensor_train, traj_tensor_train, attention_mask_train, dayofweek_train = ids_train[flag_mask], type_tensor_train[flag_mask], traj_tensor_train[flag_mask], attention_mask_train[flag_mask], dayofweek_train[flag_mask] 
        assert (ids_train == ids_test).all()
        mini_batch_size = ids_train.shape[0]
        type_tensor_train, traj_tensor_train, dayofweek_train = type_tensor_train.to(device), traj_tensor_train.to(device), dayofweek_train.to(device)
        type_tensor_test, traj_tensor_test, dayofweek_test = type_tensor_test.to(device), traj_tensor_test.to(device), dayofweek_test.to(device)

        # encode
        traj_encoding_train, type_encoding_train = encoder_embed(model, type_tensor_train, traj_tensor_train, mini_batch_size, args, normalize=if_normalize)
        traj_encoding_test, type_encoding_test = encoder_embed(model, type_tensor_test, traj_tensor_test, mini_batch_size, args, normalize=if_normalize)

        # score
        traj_encoding_by_days_train = torch.bmm(F.one_hot(dayofweek_train).permute((0,2,1)).float(), traj_encoding_train)
        traj_encoding_by_days_test = torch.bmm(F.one_hot(dayofweek_test).permute((0,2,1)).float(), traj_encoding_test)

        type_encoding_by_days_train = torch.bmm(F.one_hot(dayofweek_train).permute((0,2,1)).float(), type_encoding_train)
        type_encoding_by_days_test = torch.bmm(F.one_hot(dayofweek_test).permute((0,2,1)).float(), type_encoding_test)

        if if_normalize:
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
    #if if_sliding_window:
    #    res_name = 'b'+str(batch_size)+'l'+str(num_layers)+'d'+str(hidden_dim)+'e'+str(num_epochs)+'ws'+str(args.window_size)+'_normal'+str(int(if_normalize))+'_slidingwindow.csv'
    #else:
    #    res_name = 'b'+str(batch_size)+'l'+str(num_layers)+'d'+str(hidden_dim)+'e'+str(num_epochs)+'_normal'+str(int(if_normalize))+'.csv'
    res_name = 'anomaly_scores.csv'
    df.to_csv(os.path.join(save_dir, res_name),index=False)

def run():
    args = get_args()
    inference(args)

if __name__ == "__main__":
    run()
