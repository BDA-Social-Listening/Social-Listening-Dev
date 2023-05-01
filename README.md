# Big Data Analytics - Final Project

Team BANC

Team Members:   Balaji Jayasankar, Aniket Malsane, Nishit Jain, and Carwyn Collinsworth

Associated IDs: 114360535,         115224188,      112680897,       112605735

## Social Listening for Mental Health

The goal of this project is to distinguish between online posts which would benefit from intervention due to mental health concern, and online posts that, while potentially containing similar rhetoric, are not intended as metaphorical "cries for help".

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