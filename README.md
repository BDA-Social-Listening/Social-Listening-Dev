# Big Data Analytics - Final Project

Team BANC

Team Members:   Balaji Jayasankar, Aniket Malsane, Nishit Jain, and Carwyn Collinsworth

Associated IDs: 114360535,         115224188,      112680897,       112605735

## Social Listening for Mental Health

The goal of this project is to distinguish between online posts which would benefit from intervention due to mental health concern, and online posts that, while potentially containing similar rhetoric, are not intended as metaphorical "cries for help".

## Steps

1. First the data is scraped (via psaw) from reddit from both mental health and non-mental health communities using scrape.py and scrape_data_master.sh. This populates the data/ folder.
```
$ ./scrape_data_master.sh
```
2. A preliminary filtering is performed to remove all other attributes other than the post text and the subreddit label. This uses the filter.py script and populates the filtered_data/ folder with filtered_data.txt.
```
$ python3 filter.py data/ filtered_data/
```
3. The data is split into sentence granularity to help increase the size of our trainable data after LSH clustering. This also removes any \[removed] posts, constrains the lengths of sentences, and removes leading or trailing whitespace. This uses split.py and populates the filtered_data/ folder with  split_data.txt
```
$ python3 split.py filtered_data/
```

## Featured Media

This image represents sample data for the mental health subreddits.

<figure style="margin-left: auto;
  margin-right: auto; width: 90%; display: block;">
    <img
    src="media/sample_mental_health1.png?raw=true"
    alt="A sample of mental health sentences after processing.">
    <figcaption>Figure 1: A sample of mental health sentences after processing.</figcaption>
</figure><br>

This image shows a sample LSH for matching text from non-mental health and mental-health subreddits.

<figure style="margin-left: auto;
  margin-right: auto; width: 90%; display: block;">
    <img
    src="media/matches.jpeg?raw=true"
    alt="A sample of matches.">
    <figcaption>Figure 2: A sample of matches.</figcaption>
</figure><br>

## Setting up the environment

This project uses a Conda environment for package management. Follow the steps below to set up an environment.

1. [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html "Download and install Conda")
2. Create a Conda environment with Python 3.8.15
```
conda create -n bda_social_listening python=3.8
```
3. Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use this code.
```
conda activate bda_social_listening
```
4. Install the requirements:
```
pip3 install -r requirements.txt
```

## TODO
 - We need an efficient implementation of LSH - Aniket and Nishit (at least to reach out for information)
 - We need to implement the vectorization of text via BERT (for big data ideally, or use a lighter version roBERTa) - Carwyn and Balaji
 - LDA and (?) word cloud/cluster diagram - Aniket and Nishit
 - Add a branch to git and add the sample data from both subreddit labels - Carwyn

## Plan:
 - Take the raw text data and run it through BERT to get sentence representation vector
 - Perform LSH to cluster similar data and then use this to reduce observations
 - Perform LDA to cluster the dataset, and then use the results of this analysis to discriminate between posts in the mental health subreddit that are mental health concerns, vs not mental health focused
 - - Basically a clustering algorithm that generates clusters with "themes", and then we must give it labels based on analysis of posts in each subreddit
 - - We can use this to further discriminate between posts that show mental health concern vs discuss mental health issues but are not in need of help
 - - Potentially add this cluster to the data for the classifier (as a new feature)
 - - Get word clouds and cluster graph
 - Now we have final data to use for classifier