"""
Big Data Analytics
Team BANC
Team Members:   Balaji Jayasankar, Aniket Malsane, Nishit Jain, and Carwyn Collinsworth
Associated IDs: 114360535,         115224188,      112680897,       112605735

The purpose of this file is to pull (scrape) data from reddit via the pushshift api.

The data is created into the following directory structure:

data/
    data_SUBREDDIT_NAME/
        metadata.json
        data_0.json
        data_1.json
        data_2.json
        ...
        data_n.json
    ...

where the SUBREDDIT_NAME is a subreddit (without 'r/') such as "adhd", "anxiety", or "gaming".

To run this script, run a command such as:

```
python3 scrape.py 'adhd'
```

NOTE that this scipt automatically creates a base directory 'data1/' folder for the data.

The following source was used as base code for this file:
https://www.kaggle.com/code/nikhileswarkomati/how-to-collect-data-using-pushshift-api
"""

import json
import requests
import json
import time
import os
import sys

def main(subreddit_name, base):

    def getPushshiftData(after, before, sub):
        #Build URL
        url = 'https://api.pushshift.io/reddit/search/submission/?&size=1000&after='+str(after)+'&before='+str(before)+'&subreddit='+str(sub)
        #Request URL
        try:
            r = requests.get(url, timeout=5)
            data = json.loads(r.text)
            return data['data']
        except:
            return None    

    # Make many requests over short periods of time:

    # Start of sub: 1213305736
    start = 1223305736
    # Current date: 1682630862
    end = 1682630862
    # Approximately 25 hours
    bucket = 250000

    # init:
    data = []
    startIndex = end
    n = 0
    total_filesize = 0
    dir = "data_" + subreddit_name + "/"

    # Create directory
    if not os.path.exists(base):
        try:
            os.makedirs(base)
        except:
            print("ERROR MAKING BASE DIRECTORY at ", base, ", returning ...")

    # Enter directory
    try:
        os.chdir(base)
    except:
        print("ERROR cd'ing INTO BASE DIRECTORY at ", base, ", returning ...")

    try:
        os.mkdir(dir)
    except:
        print("ERROR CREATING SUB-DIRECTORY for subreddit at ", dir, ", skipping ...")
        return

    for i in reversed(range(start, end, bucket)):
        temp_data = getPushshiftData(i, i + bucket, subreddit_name)
        if temp_data is not None:
            data = data + temp_data

        if len(data) > 10000:
            # Write data to file
            filename = dir + 'data_' + str(n) + '.json'
            with open(filename, 'w') as f:
                json.dump(data, f)

            with open(dir + 'metadata.txt', 'a') as f:
                f.write("File: " + filename + ", " + str(i) + ", " + str(startIndex) + "\n")
            
            file_size = os.path.getsize(filename)/1000000
            total_filesize += file_size
            print("Added file ", filename, " of size ", file_size, " MB, with length: ", len(data))
            print("Total data processed: ", total_filesize, " MB")
            data = []
            
            n += 1

        time.sleep(0.2)
        print(len(data))

    print(len(data), type(data))

if __name__ == "__main__":
    reddit_name = sys.argv[1]
    base = sys.argv[2]
    print("Scraping the ", reddit_name, " subreddit into ", base)
    main(reddit_name, base)