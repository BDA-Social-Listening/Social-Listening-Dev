"""
Big Data Analytics
Team BANC
Team Members:   Balaji Jayasankar, Aniket Malsane, Nishit Jain, and Carwyn Collinsworth
Associated IDs: 114360535,         115224188,      112680897,       112605735

The purpose of this file is to load all the scraped data, and reduce it to one spark file with only selftext and subreddit attributes

The data exists in the following directory structure:

data/
    data_SUBREDDIT_NAME_1/
        metadata.json
        data_0.json
        data_1.json
        data_2.json
        ...
        data_n.json
    data_SUBREDDIT_NAME_2/
        metadata.json
        data_0.json
        data_1.json
        data_2.json
        ...
        data_n.json
    data_SUBREDDIT_NAME_3/
        metadata.json
        data_0.json
        data_1.json
        data_2.json
        ...
        data_n.json
    ...


where the SUBREDDIT_NAME_3 is a subreddit (without 'r/') such as "adhd", "anxiety", or "gaming".
"""

import sys
import json

import pyspark
from pyspark import SparkConf
from pyspark.context import SparkContext

def main():
    
    
    # sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
    conf = SparkConf().setAppName("MyApp")
    sc = SparkContext(conf=conf)
    data = sc.textFile(sys.argv[1])

    data = data.map(lambda x: json.loads(x))

if __name__ == "__main__":
    main()