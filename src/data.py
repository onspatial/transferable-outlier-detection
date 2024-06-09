import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import Dataset
import pickle
class TypeDataProcessor:
    def __init__(
        self,
        tokenizer: str,
        max_seq_len: int,
    ):
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.pad_id
        assert (
            self.pad_id is not None
        ), f"pad_id={self.pad_id} is None for tokenizer"
        self.max_seq_len = max_seq_len
        self.batch_max_len = 0
        
    def transform(self, seq):
        seq_ids = self.tokenizer.encode(seq)
        seq_len: int = len(seq_ids)
        if seq_len >= self.max_seq_len:
            input_ids = seq_ids[:self.max_seq_len]
            input_mask = [1] * self.max_seq_len
            self.batch_max_len = self.max_seq_len
        else:
            padding_len = self.max_seq_len - seq_len
            input_ids = seq_ids + [self.pad_id for _ in range(padding_len)]
            input_mask = [1] * seq_len + [0] * padding_len
            self.batch_max_len = max(seq_len, self.batch_max_len)
        outputs = [
            {
                "input_ids": torch.as_tensor(input_ids, dtype=torch.int64),
                "attention_mask": torch.as_tensor(input_mask, dtype=torch.int64),
            }
        ]
        return outputs
    
    def batch_transform(self, batch_data):
        return default_collate(batch_data)

class TrajDataProcessor:
    def __init__(
        self,
        max_seq_len: int,
    ):
        self.pad_id = 0 #TODO
        self.max_seq_len = max_seq_len
        self.batch_max_len = 0
        
    def transform(self, seq_traj):
        seq_ids = seq_traj
        seq_len: int = len(seq_ids)
        if seq_len >= self.max_seq_len:
            input_ids = seq_ids[:self.max_seq_len]
            input_mask = [1] * self.max_seq_len
            self.batch_max_len = self.max_seq_len
        else:
            padding_len = self.max_seq_len - seq_len
            input_ids = seq_ids + [[self.pad_id, self.pad_id] for _ in range(padding_len)]
            input_mask = [1] * seq_len + [0] * padding_len
            self.batch_max_len = max(seq_len, self.batch_max_len)
        outputs = [
            {
                "input_traj": torch.as_tensor(input_ids, dtype=torch.float),
                "attention_mask": torch.as_tensor(input_mask, dtype=torch.int64),
            }
        ]
        return outputs
    
    def batch_transform(self, batch_data):
        return default_collate(batch_data)
    
class TypeTokenizer:
    def __init__(self):
        self.token_ids = {
            'Apartment': 1,
            'Pub': 2,
            'Restaurant': 3,
            'Workplace': 4,
        }
        self.pad_id = 0
        
    def encode(self, seq: list):
        return [self.token_ids[item] for item in seq]


class TrajDailyDataset(Dataset):
    def __init__(self, data_path, train=True, max_daily_seq=32):
        if train:
            with open(os.path.join(data_path, f'data_dict_daily_train.pkl'), 'rb') as fp:
                self.data_dict_daily = pickle.load(fp)       
            with open(os.path.join(data_path, f'dates_train.pkl'), 'rb') as fp:
                self.dates = pickle.load(fp)
        else:
            with open(os.path.join(data_path, f'data_dict_daily_test.pkl'), 'rb') as fp:
                self.data_dict_daily = pickle.load(fp)
            with open(os.path.join(data_path, f'dates_test.pkl'), 'rb') as fp:
                self.dates = pickle.load(fp)
        self.user_ids = list(self.data_dict_daily.keys())
        tokenizer = TypeTokenizer()
        self.typeprocessor = TypeDataProcessor(tokenizer=tokenizer, max_seq_len=max_daily_seq)
        self.trajprocessor = TrajDataProcessor(max_seq_len=max_daily_seq)
        self.max_daily_seq = max_daily_seq
        
    def __len__(self) -> int: 
        return len(self.user_ids)
        
    def num_dates(self) -> int: 
        return len(self.dates)
    
    def data_to_tensor_tranform(self, data_dict):
        type_tensor_list, traj_tensor_list, attention_mask_list, dayofweek_list = [], [], [], []
        for date in self.dates:
            if date in data_dict:
                item_dict = data_dict[date]
            else:
                item_dict = {
                    'type': np.array([]),
                    'coor': np.array([[],[]]).reshape(0,2),
                    'dayofweek': pd.to_datetime(date).dayofweek
                }
                
            type_dict = self.typeprocessor.transform(item_dict['type'].tolist())[0]
            traj_dict = self.trajprocessor.transform(item_dict['coor'].tolist())[0]
            type_tensor_list.append(type_dict['input_ids'])
            traj_tensor_list.append(traj_dict['input_traj'])
            attention_mask_list.append(type_dict['attention_mask'])
            dayofweek_list.append(torch.tensor([item_dict['dayofweek']], dtype=torch.long))
        type_tensor = torch.stack(type_tensor_list)
        traj_tensor = torch.stack(traj_tensor_list)
        attention_mask = torch.stack(attention_mask_list)
        dayofweek = torch.cat(dayofweek_list)
        return type_tensor, traj_tensor, attention_mask, dayofweek
    
    def __getitem__(self, index):   #iNdex global
        ids = self.user_ids[index]
        data_dict = self.data_dict_daily[ids]
        type_tensor, traj_tensor, attention_mask, dayofweek = self.data_to_tensor_tranform(data_dict)
        return ids, type_tensor, traj_tensor, attention_mask, dayofweek

class TrajWeeklyDataset(Dataset):
    def __init__(self, data_path, train=True, max_weekly_seq=64):
        if train:
            with open(os.path.join(data_path, f'data_dict_weekly_train.pkl'), 'rb') as fp:
                self.data_dict_weekly = pickle.load(fp) 
        else:
            with open(os.path.join(data_path, f'data_dict_weekly_test.pkl'), 'rb') as fp:
                self.data_dict_weekly = pickle.load(fp)
        self.user_ids = list(self.data_dict_weekly.keys())
        self.dates = list(self.data_dict_weekly[0].keys())
        tokenizer = TypeTokenizer()
        self.typeprocessor = TypeDataProcessor(tokenizer=tokenizer, max_seq_len=max_weekly_seq)
        self.trajprocessor = TrajDataProcessor(max_seq_len=max_weekly_seq)
        self.max_weekly_seq = max_weekly_seq
        
    def __len__(self) -> int: 
        return len(self.user_ids)

    def num_weeks(self) -> int: 
        return len(self.dates)
    
    def data_to_tensor_tranform(self, data_dict):
        type_tensor_list, traj_tensor_list, attention_mask_list = [], [], []
        for date in self.dates:
            if date in data_dict:
                item_dict = data_dict[date]
            else:
                item_dict = {
                    'type': np.array([]),
                    'coor': np.array([[],[]]).reshape(0,2)
                }
                
            type_dict = self.typeprocessor.transform(item_dict['type'].tolist())[0]
            traj_dict = self.trajprocessor.transform(item_dict['coor'].tolist())[0]
            type_tensor_list.append(type_dict['input_ids'])
            traj_tensor_list.append(traj_dict['input_traj'])
            attention_mask_list.append(type_dict['attention_mask'])
        type_tensor = torch.stack(type_tensor_list)
        traj_tensor = torch.stack(traj_tensor_list)
        attention_mask = torch.stack(attention_mask_list)
        return type_tensor, traj_tensor, attention_mask
    
    def __getitem__(self, index):   #iNdex global
        ids = self.user_ids[index]
        data_dict = self.data_dict_weekly[ids]
        type_tensor, traj_tensor, attention_mask = self.data_to_tensor_tranform(data_dict)
        return ids, type_tensor, traj_tensor, attention_mask