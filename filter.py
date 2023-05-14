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

This file can be ran with the command:
```
python3 filter.py BASE_FOLDER NEWFOLDER
```
where BASE_FOLDER is something like "data/"
and NEWFOLDER is something like "filtered_data/" that is a directory that DOES NOT EXIST

```
python3 filter.py data/ filtered_data/
```

The file creates a new file filtered_data.txt
"""

import sys
import json
import os
from tqdm import tqdm

def cleanup(text):
    text = text.replace("\n", "")
    text = text.replace("'", "")
    text = text.replace('"', '')
    text = text.replace("\\", "")
    text = text.replace("\t", "")
    text = text.replace("\r", "")
    text = text.replace(",", "")
    return text

def main(data_folder, new_data_folder):

    # First get all data folders within base folder
    data_paths = []
    for subdir, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.json'):
                data_paths.append(os.path.join(subdir, file))

    # Create new directory
    try:
        os.mkdir(new_data_folder)
    except:
        print("ERROR CREATING NEW DIRECTORY at ", new_data_folder, ", returning ...")
        return
    
    total_data_list_txt = []
    
    # For each data file, let's filter
    print("FILTERING")
    for filename in tqdm(data_paths):
        # Load in file
        file_temp = open(filename, 'r')
        data_json_temp = json.load(file_temp)
        # Filter file
        data_json_temp = [el['subreddit'] + ", " + cleanup(el['selftext']) + ", " + str(el['created_utc']) + "\n" for el in data_json_temp]
        # Merge to a common list
        total_data_list_txt = total_data_list_txt + data_json_temp

    print("LENGTH: ", len(total_data_list_txt), ", needs to be > 75k")
    # print("Example: ", total_data_list_txt[0])

    # Save list as a text file
    with open(new_data_folder + "filtered_data.txt", 'w') as fout:
        print("WRITING TO FILE")
        for el in tqdm(total_data_list_txt):
            fout.write(el)

if __name__ == "__main__":
    data_folder = sys.argv[1]
    new_data_folder = sys.argv[2]
    print("Original data folder: ", data_folder, ", new data folder: ", new_data_folder)
    main(data_folder, new_data_folder)