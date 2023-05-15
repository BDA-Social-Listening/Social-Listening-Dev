# Short testing script for the eval.py true positive indices generator

import numpy as np

with open('true_pos_ind.npy', 'rb') as f:
    x = np.load(f)

    with open('filtered_data2/test_posts.csv', 'r') as csvfile:
        csvfile.readline()
        for i in range(100):
            line = csvfile.readline()
            if int(line.split(',')[0]) in x:
                print(line.split(',')[1:3])