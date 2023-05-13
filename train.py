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
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

from model import TransformerModel

def load_dataset(embed_dataset_name, labeled_dataset_name, label_map):

    batch_size = 32

    #BEGIN[ChatGPT]
    dataset = AllData(embed_dataset_name, labeled_dataset_name, label_map)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    #END[ChatGPT]

    # Create a test train validation split
    # Define the size of the splits
    train_size = 0.7  # Percentage of data for training
    val_size = 0.2  # Percentage of data for validation
    test_size = 0.1  # Percentage of data for testing

    # Calculate the size of each split
    n_samples = len(dataset)
    train_count = int(train_size * n_samples)
    val_count = int(val_size * n_samples)
    test_count = n_samples - train_count - val_count

    # Use random permutation to split the indices
    indices = np.random.permutation(n_samples)
    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count+val_count]
    test_indices = indices[train_count+val_count:]

    # Define samplers for each split
    train_sampler = data.SubsetRandomSampler(train_indices)
    val_sampler = data.SubsetRandomSampler(val_indices)
    test_sampler = data.SubsetRandomSampler(test_indices)

    # Create data loaders for each split
    train_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader

def setup_model():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    emsize = 768  # embedding dimension
    d_hid = 250  # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 3  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 4  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability
    model = TransformerModel(emsize, nhead, d_hid, nlayers, dropout).to(device)
    lr = 0.00005

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer, device

def train(dataloader, val_loader, model, criterion, optimizer, device):

    # Setup training
    model.train()
    log_interval = 200
    total_loss = 0.

    # Ensure that the labels and data have the same size
    num_batches = len(dataloader)

    # iterate over the dataloader in your training loop
    for idx, batch in tqdm(enumerate(dataloader), total=num_batches):
        batch, labels = batch

        batch = batch.to(device)
        labels = labels.to(device)

        output = model(batch)

        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if idx % log_interval == 0:
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)

            #BEGIN[ChatGPT]"add to my if idx % ... if statement [ABOVE CODE] a section which runs evalution on the val_loader for loss and accuracy"
            # Evaluation on the validation set
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            class_predictions_counts = {class_label: 0 for class_label in range(11)}
            class_label_counts = {class_label: 0 for class_label in range(11)}
            with torch.no_grad():
                for batch in val_loader:
                    batch, labels = batch
                    batch = batch.to(device)
                    labels = labels.to(device)

                    output = model(batch)
                    val_loss += criterion(output, labels).item()

                    pred = output.argmax(dim=1)
                    correct += pred.eq(labels).sum().item()
                    total += labels.size(0)

                    # fill dicts of counts
                    for class_label in range(11):
                        class_predictions_counts[class_label] += (pred == class_label).sum().item()
                        class_label_counts[class_label] += (labels == class_label).sum().item()
                    

            val_loss /= len(val_loader)
            val_acc = 100. * correct / total
            #END[ChatGPT]

            print(f'| epoch {idx:3d} | {idx:5d}/{num_batches:5d} batches | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}  | val loss {val_loss:.2f} | val acc {val_acc:.2f}%')
            #print the dictionaries
            print('Predicted Class Counts:', ', '.join([f'{k}: {"{0:<7d}".format(v)}' for k, v in class_predictions_counts.items()]))
            print('     True Class Counts:', ', '.join([f'{k}: {"{0:<7d}".format(v)}' for k, v in class_label_counts.items()]))

            total_loss = 0
            model.train()

            # make log interval less frequent
            log_interval *= 1.5

def evaluate(train_loader, test_loader):

    print("\n\nEVALUATION:\n")

    #BEGIN[ChatGPT]"add to my if idx % ... if statement [ABOVE CODE] a section which runs evalution on the val_loader for loss and accuracy"
    model.eval()
    train_loss = 0
    correct = 0
    total = 0
    class_predictions_counts = {class_label: 0 for class_label in range(11)}
    class_label_counts = {class_label: 0 for class_label in range(11)}
    with torch.no_grad():
        for batch in train_loader:
            batch, labels = batch
            batch = batch.to(device)
            labels = labels.to(device)

            output = model(batch)
            train_loss += criterion(output, labels).item()

            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

            # fill dicts of counts
            for class_label in range(11):
                class_predictions_counts[class_label] += (pred == class_label).sum().item()
                class_label_counts[class_label] += (labels == class_label).sum().item()
            

    train_loss /= len(train_loader)
    train_acc = 100. * correct / total
    #END[ChatGPT]

    print(f'{len(train_loader):5d} Batches  | Train Loss {train_loss:.2f} | Train Acc {train_acc:.2f}%')
    print('Predicted Class Counts:', ', '.join([f'{k}: {"{0:<7d}".format(v)}' for k, v in class_predictions_counts.items()]))
    print('     True Class Counts:', ', '.join([f'{k}: {"{0:<7d}".format(v)}' for k, v in class_label_counts.items()]))


    #BEGIN[ChatGPT]"add to my if idx % ... if statement [ABOVE CODE] a section which runs evalution on the val_loader for loss and accuracy"
    test_loss = 0
    correct = 0
    total = 0
    class_predictions_counts = {class_label: 0 for class_label in range(11)}
    class_label_counts = {class_label: 0 for class_label in range(11)}
    with torch.no_grad():
        for batch in test_loader:
            batch, labels = batch
            batch = batch.to(device)
            labels = labels.to(device)

            output = model(batch)
            test_loss += criterion(output, labels).item()

            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

            # fill dicts of counts
            for class_label in range(11):
                class_predictions_counts[class_label] += (pred == class_label).sum().item()
                class_label_counts[class_label] += (labels == class_label).sum().item()
            

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    #END[ChatGPT]

    print(f'{len(test_loader):5d} Batches  | Test Loss {test_loss:.2f} | Test Acc {test_acc:.2f}%')
    print('Predicted Class Counts:', ', '.join([f'{k}: {"{0:<7d}".format(v)}' for k, v in class_predictions_counts.items()]))
    print('     True Class Counts:', ', '.join([f'{k}: {"{0:<7d}".format(v)}' for k, v in class_label_counts.items()]))
    
class AllData(Dataset):
    def __init__(self, path, label_path, label_map):
        self.data = open(path, 'r').readlines()
        self.label_map = label_map
        self.labels = open(label_path, 'r').readlines()
        
    def __len__(self):
        assert len(self.data) ==  len(self.labels)
        return len(self.data)
    
    def __getitem__(self, index):
        # num_classes = 11
        class_data = torch.tensor(int(self.label_map[self.labels[index].strip().split(',')[0]]))
        data =  torch.tensor(np.fromstring(self.data[index].strip()[1:-1], sep=' ').astype(np.float32))
        return (data, class_data)

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

    train_loader, val_loader, test_loader = load_dataset(embed_dataset_name, labeled_dataset_name, label_map)

    model, criterion, optimizer, device = setup_model()

    try:
        model.load_state_dict(torch.load("models/fully_trained_transformer_model"))
    except:
        train(train_loader, val_loader, model, criterion, optimizer, device)

    torch.save(model.state_dict(), "models/fully_trained_transformer_model")
    evaluate(train_loader, test_loader)

    