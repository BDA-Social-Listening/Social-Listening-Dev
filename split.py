"""
Big Data Analytics
Team BANC
Team Members:   Balaji Jayasankar, Aniket Malsane, Nishit Jain, and Carwyn Collinsworth
Associated IDs: 114360535,         115224188,      112680897,       112605735

This file takes the filtered_data.txt file, which is in the form of a list of posts represented as dictionaries with the subreddit and selftext attributes remaining after the filter.
This file breaks up all posts into sentence level representations, removing posts that are too long or short.
This should help improve the number of posts that remain trainable after LSH.

Sample command to run:
```
$ python3 split.py filtered_data/
```
or
```
$ python3 split.py [FOLDER_WITH_filtered_data.txt]
```
"""

import sys
import pyspark
from pyspark import SparkConf
from pyspark.context import SparkContext

def main(data_folder):
    # sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
    conf = SparkConf().setAppName("MyApp")
    sc = SparkContext(conf=conf)
    data = sc.textFile(data_folder + "filtered_data.txt")

    # Format as a key value pair from ?one long string?
    data = data.map(lambda x: (x.split(',')[0], ''.join(x.split(',')[1:])))

    # Remove any data that is "[removed]"
    data = data.filter(lambda x: "[removed]" not in x[1])

    # Split into sentences
    def splitSentences(record):
        split = record[1].split('.')
        for sentence in split:
            yield (record[0], sentence.strip())

    data = data.flatMap(lambda x: splitSentences(x))

    # Remove any records that are super short or super long
    data = data.filter(lambda x: len(x[1]) > 10 and len(x[1]) < 250)

    # Save data as text file
    data.saveAsTextFile(data_folder + "split_data")

if __name__ == "__main__":
    data_folder = sys.argv[1]
    print("Data folder: ", data_folder)
    main(data_folder)