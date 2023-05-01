# import praw
import pandas as pd
import json
from IPython.display import display

# Read in secrets:
f = open('secret.json',)
v = json.load(f)

# SOURCE:
# https://infatica.io/blog/scraping-reddit-with-scraper-api/
"""
# Read-only instance
reddit_read_only = praw.Reddit(client_id=v["client_id"],         # your client id
                               client_secret=v["client_secret"],    # your client secret
                               user_agent=v["user_agent"])   # your user agent


subreddit = reddit_read_only.subreddit("mentalhealth")

posts = subreddit.top("month")
# Scraping the top posts of the current month

posts_dict = {"Title": [], "Post Text": [],
              "ID": [], "Score": [],
              "Total Comments": [], "Post URL": []
              }

for post in posts:
    # Title of each post
    posts_dict["Title"].append(post.title)

    # Text inside a post
    posts_dict["Post Text"].append(post.selftext)

    # Unique ID of each post
    posts_dict["ID"].append(post.id)

    # The score of a post
    posts_dict["Score"].append(post.score)

    # Total number of comments inside the post
    posts_dict["Total Comments"].append(post.num_comments)

    # URL of each post
    posts_dict["Post URL"].append(post.url)

# Saving the data in a pandas dataframe
top_posts = pd.DataFrame(posts_dict)

display(top_posts)
"""

# SOURCE:
# https://www.kaggle.com/code/nikhileswarkomati/how-to-collect-data-using-pushshift-api

import pandas as pd
import requests #Pushshift accesses Reddit via an url so this is needed
import json #JSON manipulation
import csv #To Convert final table into a csv file to save to your machine
import time
import datetime
import time
import os

def getPushshiftData(after, before, sub):
    #Build URL
    url = 'https://api.pushshift.io/reddit/search/submission/?&size=1000&after='+str(after)+'&before='+str(before)+'&subreddit='+str(sub)
    #Print URL to show user
    # print(url)
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
dir = "data_music/"

for i in reversed(range(start, end, bucket)):
    temp_data = getPushshiftData(i, i + bucket, 'music')
    if temp_data is not None:
        data = data + temp_data

    if len(data) > 10000:
        # Write data to file
        filename = dir + 'data_' + str(n) + '.json'
        with open(filename, 'w') as f:
            json.dump(data, f)

        with open(dir + 'metadata.json', 'a') as f:
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