"""
Big Data Analytics
Team BANC
Team Members:   Balaji Jayasankar, Aniket Malsane, Nishit Jain, and Carwyn Collinsworth
Associated IDs: 114360535,         115224188,      112680897,       112605735

This file creates data for the confusion matrix, true positive indeces (for LDA).
This file is also used to generate archetype examples (indecs, and test.py for text).

Run with
```
$python3 eval.py
```
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from model import LogisticModel

"""
Dataset Class for creating PyTorch Dataloader
Modified with ChatGPT
"""
class AllData(Dataset):
    def __init__(self, path, label_path, label_map):
        self.data = open(path, 'r').readlines()
        self.label_map = label_map
        self.labels = {}

        # Read the labels and store them in a dictionary
        with open(label_path, 'r') as f:
            f.readline()
            # header = f.readline().strip().split(',')
            # index_col = header.index('index')
            # label_col = header.index('label')
            for line in f:
                row = line.strip().split(',')
                index = int(row[0])
                label = int(label_map[row[1]])
                self.labels[index] = label
        
        # Check that the data and label files have the same number of rows
        assert len(self.data) == len(self.labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Load the data, index, and class label
        row = self.data[index].strip().split(',')
        indexid = int(row[0])
        data = np.fromstring(row[1].strip()[1:-1], sep=' ')
        data = torch.tensor(data.astype(np.float32))
        class_data = torch.tensor(self.labels[indexid])
        
        return (data, class_data, indexid)
    
label_map = {
    "adhd":0,
    "anxiety":0,
    "depression":0,
    "mentalhealth":0,
    "mentalillness":0,
    "socialanxiety":0,
    "suicidewatch":0,
    "gaming":1,
    "guns":1,
    "music":1,
    "parenting":1
}

# Load in pretrained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

emsize = 768
dropout = 0.2
classes = 2
batch_size = 64
model = LogisticModel(emsize, classes, dropout).to(device)

model.load_state_dict(torch.load("models/fully_trained_logistic_binary_model"))

# Load test dataset
labeled_dataset_name = 'filtered_data2/test_posts.csv'
embed_dataset_name = 'filtered_data2/test_embeddings.txt'
dataset = AllData(embed_dataset_name, labeled_dataset_name, label_map)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, pin_memory=True)

# Get stats from model:
true_positive_indeces = []
z = 0

correct = 0
total = 0

tp = 0
fp = 0
fn = 0
tn = 0

with torch.no_grad():
    for batch in dataloader:
        batch, labels, index = batch
        batch = batch.to(device)
        labels = labels.to(device)

        output = model(batch)

        pred = output.argmax(dim=1).to(labels.device)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

        # Confusion matrix data
        #BEGIN[ChatGPT]
        # Calculate true positives, false positives, false negatives, and true negatives
        tp += ((pred == 0) & (labels == 0)).sum().item()
        fp += ((pred == 0) & (labels == 1)).sum().item()
        fn += ((pred == 1) & (labels == 0)).sum().item()
        tn += ((pred == 1) & (labels == 1)).sum().item()
        #END[ChatGPT]

        # Add the correct indeces to the list
        indices = np.where((pred.cpu() == 0) & (labels.cpu() == 0)) # CHANGE THIS LINE TO CHANGE THE TYPE OF INDECES WE WANT
        result = index[indices]
        result = list(result.numpy())
        true_positive_indeces = true_positive_indeces + result

        if z % 100000 == 0 and z > 0:
            print(z)
            # break
        z += 1


print("Number Correct: ", correct, ", Total: ", total)
print("# True positives: ", len(true_positive_indeces), ", examples: ", true_positive_indeces[0:4], " ...")
print("tp: ", tp, ", fp: ", fp, ", tn: ", tn, ", fn: ", fn)
print("Accuracy: ", (tp+tn)/(tp+tn+fp+fn), ", or ", correct/total)

with open('true_pos_ind.npy', 'wb') as f:
    np.save(f, true_positive_indeces)