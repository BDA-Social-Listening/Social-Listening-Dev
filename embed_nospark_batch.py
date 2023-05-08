"""
Big Data Analytics
Team BANC
Team Members:   Balaji Jayasankar, Aniket Malsane, Nishit Jain, and Carwyn Collinsworth
Associated IDs: 114360535,         115224188,      112680897,       112605735

This file takes the split_data file (folder), which contains all of the filtered, cleaned data, and runs every record through BERT to get sentence embeddings.
This file saves the data into a new folder, "embed_data"

Loads data from "split_data_single/part-00000.txt"

NOTE: This file uses pyspark and individual records. Therefore, no batching occurs, and the file is read into a cluster. Also likely CPU.

Sample command to run:
```
$ python3 embed_nospark_batch.py filtered_data/split_data_single/part-00000 filtered_data/
```
or
```
$ python3 embed_nospark_batch.py [FILE] [FOLDER_TO_SAVE_EMBED_DATA]
```

This takes 6600 Seconds and takes up 14GB
"""

import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer

def main(data_file, data_folder):

    #BEGIN[ChatGPT]"How can I get sentence embeddings with some BERT (maybe roberta) implementation locally" ... "I have code that takes a sentence and gets a sentence embedding with DistilBERT. How do I generalize this function to efficiently take a list of sentences, and return their sentence embeddings. Note, this should run on the GPU: "def get_embedding_from_raw_sentence(sentence):encoded_sentence = tokenizer.encode(sentence, add_special_tokens=True, max_length=max_length, truncation=True)input_ids = torch.tensor(encoded_sentence).unsqueeze(0)  # Batch size 1outputs = model(input_ids)sentence_embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings"
    max_length=128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base', max_length=max_length) # max length of 512
    # model = RobertaModel.from_pretrained('roberta-base')

    # USING ALBERT, IT TAKES ABOUT 27:19 MINUTES:SECONDS FOR 1000 PREDICTIONS (note this had 512 as tokenizer.encode parameter, 128 otherwise) (eta 18.75 days)
    # model = AlbertModel.from_pretrained('albert-base-v2', max_length=max_length)
    # tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', max_length=max_length)

    # USING DistilBERT, IT TAKES ABOUT 11:53 MINUTES:SECONDS FOR 1000 PREDICTIONS (note this had 128 as tokenizer.encode parameter) (eta 8.3 days)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased', max_length=max_length)
    model = DistilBertModel.from_pretrained('distilbert-base-cased').to(device)

    def get_embeddings_from_raw_sentences(sentences, model, tokenizer, device):
        encoded_sentences = tokenizer.batch_encode_plus(sentences, add_special_tokens=True, max_length=max_length, truncation=True, padding=True)
        input_ids = torch.tensor(encoded_sentences['input_ids']).to(device)
        attention_mask = torch.tensor(encoded_sentences['attention_mask']).to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        sentence_embeddings = outputs.last_hidden_state.mean(dim=1)
        return sentence_embeddings

    #END[ChatGPT]

    num_lines = sum(1 for _ in open(data_file,'r'))
    print("Number of data samples: ", num_lines)

    start_time = time.time()

    # Load data in batches
    with open(data_file, 'r') as f:
        buf = []
        for i in tqdm(range(num_lines)):
            line = f.readline()
            subreddit, selftext = line.split(",", 1)
            subreddit, selftext = subreddit.strip(), selftext.strip()

            # Check if buffer is full
            if len(buf) == 32 or i == num_lines: # If this line errors, add a -1 after 'num_lines'
                # Perform encoding to sentence representations
                sentence_embeddings = get_embeddings_from_raw_sentences(list(list(zip(*buf))[1]), model, tokenizer, device)
                sentence_embeddings = sentence_embeddings.detach().cpu().numpy()

                # Save to disk
                with open(data_folder + "embed_data.txt", "a") as fsave:  # "a" for append!
                    for emb in sentence_embeddings:
                        fsave.write(np.array_str(emb, max_line_width=1000000) + "\n")

                # Clear buffer
                buf = []
            
            buf.append((subreddit, selftext))

    print("Time taken: {} seconds".format(time.time() - start_time))

if __name__ == "__main__":
    data_file = sys.argv[1]
    data_folder = sys.argv[2]
    print("Data folder: ", data_folder, ", Data file: ", data_file)
    main(data_file, data_folder)