#!/usr/bin/env python3

# python code that reads the csv files from a folder, removes the first row, and then saves them again with the same name

import os
import pandas as pd

# arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", 
                    help="folder with csv files",
                    required=True)
args = parser.parse_args()

# get all files in the folder
files = os.listdir(args.folder)

# loop over all files
for file in files:
    if file.endswith('.csv'):
        # read the file
        df = pd.read_csv(args.folder + "/" + file)
        # remove the first row
        df = df.iloc[1:]
        # save the file again
        df.to_csv(args.folder + "/" + file, index=False)
    else:
        pass
