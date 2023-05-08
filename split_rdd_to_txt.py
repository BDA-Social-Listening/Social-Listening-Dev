"""
Big Data Analytics
Team BANC
Team Members:   Balaji Jayasankar, Aniket Malsane, Nishit Jain, and Carwyn Collinsworth
Associated IDs: 114360535,         115224188,      112680897,       112605735

This file takes the data saved in split_data and reorganizes it into one single file. This file will be:
split_data_single/part-00000

This file is ran with
```
$ python3 split_rdd_to_txt.py
```
"""

import pyspark
from pyspark import SparkConf
from pyspark.context import SparkContext

df = "filtered_data/"

conf = SparkConf().setAppName("MyApp")
sc = SparkContext(conf=conf)
data = sc.textFile(df + "split_data")

data2 = data.coalesce(1, shuffle = True).saveAsTextFile(df + "split_data_single")