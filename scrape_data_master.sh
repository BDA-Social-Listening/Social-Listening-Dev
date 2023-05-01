# Big Data Analytics
# Team BANC
# Team Members:   Balaji Jayasankar, Aniket Malsane, Nishit Jain, and Carwyn Collinsworth
# Associated IDs: 114360535,         115224188,      112680897,       112605735

# This is a shell script to automate data 'scraping' from reddit using the 'scrape.py' file.

# Modify the list to generate data for different subreddits
# Modify the base variable to change the base directory to be created

# NOTE THAT RUNNING THIS FILE AS-IS WILL PROCEED TO ALLOCATE 17+GB ON YOUR DISK IN A MATTER OF A FEW HOURS!

#!/bin/sh

base='data1/'
list='adhd anxiety depression gaming guns mentalhealth mentalillness music parenting socialanxiety suicidewatch'

for i in $list; do
    python3 scrape.py $i $base
done