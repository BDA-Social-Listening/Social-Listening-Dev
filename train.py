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

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

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
    pass

def train(dataloader, labelloader):

    # Ensure that the labels and data have the same size
    num_batches = len(dataloader)
    print(num_batches, len(labelloader))
    assert num_batches == len(labelloader)

    label_loader_enum = iter(labelloader)

    # iterate over the dataloader in your training loop
    for idx, batch in tqdm(enumerate(dataloader), total=num_batches):
        labels = next(label_loader_enum)


#BEGIN[ChatGPT]"I have data in a nonstandard format - a string that represents an array for each record. The data is too big to fit in memory. How do I use the pytorch dataloader to efficiently train on this data"
class RedditDataset(Dataset):
    def __init__(self, path):
        self.data = open(path, 'r').readlines()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        #END[ChatGPT]
        return torch.tensor(np.fromstring(self.data[index].strip()[1:-1], sep=' '))
#END[ChatGPT]

class RedditLabels(Dataset):
    def __init__(self, path, label_map):
        self.data = open(path, 'r').readlines()
        self.label_map = label_map
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.tensor(int(self.label_map[self.data[index].strip().split(',')[0]]))

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

    train(dataloader, labelloader)