"""
Big Data Analytics
Team BANC
Team Members:   Balaji Jayasankar, Aniket Malsane, Nishit Jain, and Carwyn Collinsworth
Associated IDs: 114360535,         115224188,      112680897,       112605735

This file trains a model to predict the subreddit title from the text embedding (DistilBert)

Sample command to run:
```
$ python3 train.py
```
"""
import time
import math
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model import TransformerModel

def load_dataset(embed_dataset_name, labeled_dataset_name, label_map):
    #BEGIN[ChatGPT]
    # create an instance of the custom dataset
    dataset = RedditDataset(embed_dataset_name)

    # create a dataloader to load data in batches (KEEP shuffle=False or the code breaks)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)
    #END[ChatGPT]

    # Load the associated labels from the singular split data file
    labelset = RedditLabels(labeled_dataset_name, label_map)

    # create a dataloader to load data in batches (KEEP shuffle=False or the code breaks)
    labelloader = DataLoader(labelset, batch_size=32, shuffle=False, pin_memory=True)

    # return the dataloader and associated labels
    return dataloader, labelloader

def setup_model():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    emsize = 768  # embedding dimension
    d_hid = 250  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 3  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 4  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability
    model = TransformerModel(emsize, nhead, d_hid, nlayers, dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 0.001  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer, device

def train(dataloader, labelloader, model, criterion, optimizer, device):

    # Setup training
    model.train()
    log_interval = 200
    total_loss = 0.

    # Ensure that the labels and data have the same size
    num_batches = len(dataloader)
    print(num_batches, len(labelloader))
    assert num_batches == len(labelloader)

    label_loader_enum = iter(labelloader)

    # iterate over the dataloader in your training loop
    for idx, batch in tqdm(enumerate(dataloader), total=num_batches):
        labels = next(label_loader_enum)

        batch = batch.to(device)
        labels = labels.to(device)

        output = model(batch)

        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if idx % log_interval == 0 and idx > 0:
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            # print(f'| epoch {idx:3d} | {idx:5d}/{num_batches:5d} batches | '
            #       f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            print(output)
            total_loss = 0

#BEGIN[ChatGPT]"I have data in a nonstandard format - a string that represents an array for each record. The data is too big to fit in memory. How do I use the pytorch dataloader to efficiently train on this data"
class RedditDataset(Dataset):
    def __init__(self, path):
        self.data = open(path, 'r').readlines()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        #END[ChatGPT]
        return torch.tensor(np.fromstring(self.data[index].strip()[1:-1], sep=' ').astype(np.float32))
#END[ChatGPT]

class RedditLabels(Dataset):
    def __init__(self, path, label_map):
        self.data = open(path, 'r').readlines()
        self.label_map = label_map
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        num_classes = 11
        val = int(self.label_map[self.data[index].strip().split(',')[0]])
        one_hot_matrix = np.eye(num_classes)[val]
        return torch.tensor(one_hot_matrix)

if __name__ == "__main__":
    embed_dataset_name = 'filtered_data/embed_data.txt'
    labeled_dataset_name = 'filtered_data/split_data_single/part-00000'

    label_map = {
        "adhd":0,
        "anxiety":1,
        "depression":2,
        "mentalhealth":3,
        "mentalillness":4,
        "socialanxiety":5,
        "suicidewatch":6,
        "gaming":7,
        "guns":8,
        "music":9,
        "parenting":10
    }

    dataloader, labelloader = load_dataset(embed_dataset_name, labeled_dataset_name, label_map)

    model, criterion, optimizer, device = setup_model()

    train(dataloader, labelloader, model, criterion, optimizer, device)