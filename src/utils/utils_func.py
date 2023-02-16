import torch
import os
import pickle
import pandas as pd



"""
common utils
"""
def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
                

def make_dataset_path(base_path):
    dataset_path = {}
    for split in ['train', 'val', 'test']:
        dataset_path[split] = base_path+'data/kowikitext/processed/kowikitext.'+ split + '.pkl'
    return dataset_path


def save_checkpoint(file, model, optimizer):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, file)
    print('model pt file is being saved\n')