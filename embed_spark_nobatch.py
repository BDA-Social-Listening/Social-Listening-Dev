"""
Big Data Analytics
Team BANC
Team Members:   Balaji Jayasankar, Aniket Malsane, Nishit Jain, and Carwyn Collinsworth
Associated IDs: 114360535,         115224188,      112680897,       112605735

This file takes the split_data file (folder), which contains all of the filtered, cleaned data, and runs every record through BERT to get sentence embeddings.
This file saves the data into a new folder, "embed_data"

NOTE: This file uses pyspark and individual records. Therefore, no batching occurs, and the file is read into a cluster. Also likely CPU.

Sample command to run:
```
$ python3 embed_spark_nobatch.py filtered_data/
```
or
```
$ python3 embed_spark_nobatch.py [FOLDER_WITH_split_data]
```
"""

import sys
import time
import torch
import pyspark
from pyspark import SparkConf
from pyspark.context import SparkContext
from transformers import RobertaModel, RobertaTokenizer
from transformers import AlbertModel, AlbertTokenizer
from transformers import DistilBertModel, DistilBertTokenizer

def main(data_folder):
    conf = SparkConf().setAppName("MyApp")
    sc = SparkContext(conf=conf)
    data = sc.textFile(data_folder + "split_data")

    #BEGIN[ChatGPT]"from a sentence such as "mentalhealth, Truck driver substance abuse has been a growing problem in the United States for many years. The DOT has implemented many policies in an attempt to address this issue, but t  The job can be extremely stressfu", use a rdd.map to extract the label (the word before the first comma) and the rest of the text as separate elements"
    data = data.map(lambda line: line.split(",", 1))
    data = data.filter(lambda line: len(line) == 2)
    data = data.map(lambda parts: (parts[0].strip(), parts[1].strip()))
    #END[ChatGPT]

    print("Number of data samples: ", data.count()) # about 1.3 million
    fraction = 1000 / float(data.count())
    data = data.sample(False, fraction)
    print("Number of data samples: ", data.count()) # new count

    #BEGIN[ChatGPT]"How can I get sentence embeddings with some BERT (maybe roberta) implementation locally"

    max_length=128

    start_time = time.time()
    # USING RoBERTa, IT TAKES ABOUT 20:30 MINUTES:SECONDS FOR 1000 PREDICTIONS (note this had 128 as tokenizer.encode parameter) (eta 14.24 days)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', max_length=max_length)
    model = RobertaModel.from_pretrained('roberta-base')

    # USING ALBERT, IT TAKES ABOUT 27:19 MINUTES:SECONDS FOR 1000 PREDICTIONS (note this had 512 as tokenizer.encode parameter, 128 otherwise) (eta 18.75 days)
    # model = AlbertModel.from_pretrained('albert-base-v2', max_length=max_length)
    # tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', max_length=max_length)

    # USING DistilBERT, IT TAKES ABOUT 11:53 MINUTES:SECONDS FOR 1000 PREDICTIONS (note this had 128 as tokenizer.encode parameter) (eta 8.33 days)
    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased', max_length=max_length)
    # model = DistilBertModel.from_pretrained('distilbert-base-cased')

    def get_embedding_from_raw_sentence(sentence):
        encoded_sentence = tokenizer.encode(sentence, add_special_tokens=True, max_length=max_length, truncation=True)
        input_ids = torch.tensor(encoded_sentence).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings
        return sentence_embedding
    #END[ChatGPT]

    data = data.map(lambda parts: (parts[0], get_embedding_from_raw_sentence(parts[1])))

    # Save data as text file
    data.saveAsTextFile(data_folder + "embed_data")

    print("Time taken: {} seconds".format(time.time() - start_time))

    # print(data.take(3))

if __name__ == "__main__":
    data_folder = sys.argv[1]
    print("Data folder: ", data_folder)
    main(data_folder)