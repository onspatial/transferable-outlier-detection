#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--save_path', type=str, default='./models')
    parser.add_argument('--encoder_type', type=str, default='transformer')
    parser.add_argument('--if_normalize', type=bool, default=True)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--max_daily_seq', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--input_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay_step_size', type=int, default=50)
    parser.add_argument('--weight_decay', type=float, default=0.9)
    parser.add_argument('--is_baseline', type=str, default='False')
    parser.add_argument('--is_transfer', type=str, default='False')
    parser.add_argument('--method', type=str, default='trad')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    data_path = args.data_path
    save_path = args.save_path
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    max_daily_seq = args.max_daily_seq
    input_dim, hidden_dim, output_dim = args.input_dim, args.hidden_dim, args.output_dim
    num_layers = args.num_layers
    encoder_flag = args.encoder_type
    if_normalize = args.if_normalize
    dataset_name = args.data_path.split('/')[-1]
    if args.is_baseline == 'False':
        folder_name = 'b'+str(batch_size)+'l'+str(num_layers)+'d'+str(hidden_dim)+'e'+str(num_epochs)+'_normal'+str(int(if_normalize))
        save_dir = os.path.join(args.save_path, dataset_name, args.encoder_type, folder_name)
    if args.is_transfer == 'True':
        print('i am here')
        save_dir = os.path.join(args.save_path, dataset_name, args.encoder_type)
    if args.is_baseline == 'True':
        save_dir = os.path.join(args.save_path, dataset_name, args.method)
    
    if args.is_transfer == 'True':
        res_name = 'transfer_anomaly_scores.csv'
    else:
        res_name = 'anomaly_scores.csv'
    print(os.path.join(save_dir, res_name))
    print(os.path.join(data_path, 'groundtruth.csv'))

    if not os.path.exists(os.path.join(save_dir, res_name)):
        raise Exception(f'{os.path.join(save_dir, res_name)} does not exist.')
    df = pd.read_csv(os.path.join(save_dir, res_name))
    df['TrajScore'] = 1 - df['TrajScore'] 
    df['TypeScore'] = 1 - df['TypeScore'] 
    #df['AllScore'] = 1 - df['AllScore']
    df['AllScore'] = df[['TrajScore','TypeScore']].max(axis=1)
    groundtruth = pd.read_csv(os.path.join(data_path, 'groundtruth.csv'))
    #print(groundtruth.columns)
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


    if args.is_transfer != 'True':
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





