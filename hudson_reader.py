#!/usr/bin/env python3

"""
Python code to read the output files from the Hudson OD reader from our lab.
Each file contains the OD readings for a single plate, and the file name reflects
the wavelength of the OD reading, the year, the month, and the day, then the time 
of the reading after an underscore (HHMMSS).
They look like: OD 600 Hudson_20230306_170344.csv

Every file consists on only 1 reading, so depending on the number of plates tested, 
there will be a different number of files. They always preserve the same order, so 
the first file will always be the first plate, the second file the second plate, etc.
The number of plates tested will be passed as an argument to the script.

The script will read a Design file with some custom metadata for each plate, and
the plate number, which will be match the order of the files. Each plate will have a
pattern file, which will be used to match the wells to the metadata.

The script will first read a folder with the files and create a dataframe with the following columns:
    1. File name
    2. Plate number
    3. Well
    4. Time
    5. OD reading

Then it will read the Design file and join the dataframe by plate number.

Lastly, it will read the pattern file and join the dataframe by well.

The timeseries will be smoothed using a Wiener filter, and the AUC will be calculated
using the trapezoidal rule. Then the AUCs will be plotted as heatmaps per plate, and the
dataframe will be saved as a csv file with the name "Summary.csv". The timeseries will be 
plotted per plate as lineplots in pdf, and the dataframe saved as a csv file with the name 
of "Timeseries.csv". Both files will have all the previous metadata.

The python script takes the following parameters:
    -i <input Design file>
    -f <input folder where the csv files are located>
    -o <output folder name>
    -p <number of plates tested> 
"""

import sys, getopt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import wiener
# from matplotlib.backends.backend_pdf import PdfPages
import argparse
from tqdm import tqdm
import multiprocessing as mp
from datetime import datetime
import re
from itertools import product


__author__ = 'Daniel Martínez Martínez'
__copyright__ = 'Copyright (C) 2023 Daniel Martínez Martínez'
__license__ = 'MIT License'
__email__ = 'dmartimarti **AT** gmail.com'
__maintainer__ = 'Daniel Martínez Martínez'
__status__ = 'alpha'
__date__ = 'Jun 2023'
__version__ = '0.1'

# arguments
parser = argparse.ArgumentParser(description='Reads the output files from the Hudson OD reader from our lab.')
parser.add_argument('-i', 
                    '--input', 
                    help='Input Design file', 
                    default='Design.xlsx',
                    required=True)
parser.add_argument('-f', 
                    '--folder', 
                    help='Input folder where the csv files are located', 
                    required=True)
parser.add_argument('-o', 
                    '--output', 
                    help='Output folder name', 
                    required=True)
parser.add_argument('-p', 
                    '--plates', 
                    help='Number of plates tested', 
                    required=True)
parser.add_argument('-t',
                    '--threads',
                    default=1,
                    help='Number of threads to use',
                    required=False)
args = parser.parse_args()

# class of colors to print in the terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# initialise variables
ROOT = args.folder
OUTPUT = args.output
OUTPUT_PLOTS = os.path.join(OUTPUT, 'plots')


# functions of the script
def list_files(folder):
    """ 
    Function that lists all the files in a folder and returns a list with the 
    file names and a vector with the times of the readings.
    """

    # list all csv files in the folder
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    files.sort()

    # from the names, if there is a name shorter than the others, remove it
    min_len = min([len(f) for f in files])
    max_len = max([len(f) for f in files])

    if min_len != max_len:
        print('Removing file(s) with shorter names')
        files = [f for f in files if len(f) != min_len]
    
    return files


def get_sheet_names(file):
    """
    Function that gets the sheet names from an Excel file
    """
    xlfile = pd.ExcelFile(file)
    # sheet names if they don't start by '_'
    sheet_names = [sheet for sheet in xlfile.sheet_names if sheet != '_']
    return sheet_names


def get_plate_size(df):
    """
    Function that gets the plate size from the Well column.
    """
    
    if "P1" in df["Well"].unique():
        return 384
    else:
        return 96



def get_dates(files):
    """
    Function that gets the dates from the files.
    """

    # get the dates from files
    dates = [datetime.strptime(re.findall(r'\d{6}_\d{6}', f)[0], '%y%m%d_%H%M%S') for f in files]
    return dates

def interval_time(dates, plates):
    """
    Function that calculates the interval time between each reading.
    """

    unique_dates = dates[::plates]
    times = [0]
    for i in range(len(unique_dates)-1):
        times.append((unique_dates[i+1] - unique_dates[i]).seconds)

    # average of the times
    avg_time = np.mean(times)
    # from the average, calculate a round hours interval
    interval = round(avg_time/60/60, 2)
    return interval

def hudson_df_reader_parallel(args):
    """ 
    Function that reads a csv file and returns a dataframe 
    with the following columns: ['well', 'OD', 'Time', 'Plate'] 
    """
    file, time, plate = args
    
    df =  pd.read_csv(file, header=None)
    df = df.iloc[1:]
    # if the plate is a 96-well plate insert the letters A-H and the numbers 1-12,
    # if it is a 384-well plate insert the letters A-P and the numbers 1-24
    if df.shape[0] == 96:
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        numbers = [str(i) for i in range(1,13)]
    elif df.shape[0] == 384:
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J','K','L','M','N','O','P']
        numbers = [str(i) for i in range(1,25)]
    else:
        print('Plate size not supported')
        sys.exit(2)

    #insert the letters and numbers into the dataframe as the Well column
    df['Well'] = [l + n for l in letters for n in numbers]

    # create a list with all the wells

    df['Date'] = time
    df['Plate'] = plate
    df.columns = ['OD_raw', 'Well', 'Date', 'Plate']
    return df

def wiener_filter(df, interval):
    """
    Function that applies the wiener filter to a dataframe with the OD values
    and returns a dataframe with the filtered values.
    """
    
    # apply a Wiener function to the OD values per plate and well
    filtered = df.groupby(['Plate', 'Well'])['OD_raw'].apply(lambda x: wiener(x.to_numpy(), 5))
    filtered = filtered.to_frame().reset_index()
    filtered = filtered.explode('OD_raw')
    filtered['Time_h'] = filtered.groupby(['Plate', 'Well']).cumcount().to_numpy() * interval
    filtered = filtered.reset_index(drop=True)
    filtered = filtered.sort_values(by=['Plate', 'Well'])
    filtered.columns = ['Plate', 'Well', 'OD_w', 'Time_h']

    final_df_w = df.merge(filtered, on=['Plate', 'Well', 'Time_h'], how='left')
    return final_df_w


def plot_plate(df, plate, save_path=None):
    """
    Function that plots the OD values of a plate.
    """
    plt.figure(figsize=(25,17))
    df = df[df['Plate'] == plate]

    for i, well in enumerate(df['Well'].unique()):
        plt.subplot(8,12,i+1)
        # filter the data for the well
        well_data = df[df['Well'] == well]
        # plot the data
        plt.plot(well_data['Time_h'], well_data['OD_w'])
        # if the well is not in the last row, remove the xticks
        if well[0] != 'H':
            plt.xticks([])
        # if the well is in the first column, set the y ticks
        if well[1] != '1' and well[1] not in ['10', '11', '12']:
            plt.yticks([])
        # set the title
        plt.title(well)
    plt.suptitle(f'Plate {plate}')
    # save the plot
    plt.savefig(f'{save_path}/plate_{plate}.pdf')
    plt.close()


def plot_plate_wrapper(args):
    """
    Wrapper function for the plot_plate function.
    """
    plot_plate(*args)


def get_well_names(num_letters, num_numbers):
    """
    Function that returns a list of all possible combinations of letters (from A to a specified number) and numbers (from 1 to a specified number)
    Parameters
    ----------
    num_letters : int
        The number of letters to be used
    num_numbers : int
        The number of numbers to be used
    Returns
    -------
    list
        A list of all possible combinations of letters (from A to a specified number) and numbers (from 1 to a specified number)
    """
    # get the letters
    letters = [chr(i) for i in range(65, 65+num_letters)]
    # get the numbers
    numbers = [str(i) for i in range(1, num_numbers+1)]
    # get all the combinations of letters and numbers
    well_names = [i+j for i, j in product(letters, numbers)]
    return well_names


def calculate_aucs(df):
    """
    Function that calculates the AUCs for each well in a plate.
    """
    # make the df wider so each row is a well
    df_wide = df.pivot_table(index=['Plate', 'Well'], columns='Time_h', values='OD_w').reset_index()
    # calculate the AUCs by row
    df_wide['AUC'] = df_wide.apply(lambda x: np.trapz(x[2:-1]), axis=1)
    # return the df with the AUCs
    df_wide = df_wide[['Plate', 'Well', 'AUC']]
    # calculate log2 of the AUCs
    with np.errstate(divide='ignore', invalid='ignore'):
        df_wide.loc[:, 'AUC_log2'] = np.log2(df_wide['AUC'])
    return df_wide


def get_pattern_metadata(pattern, plate_size=96):
    """
    Function that returns the pattern metadata from a pattern file
    """
    pattern_vars = get_sheet_names(pattern)
    pattern_df = pd.DataFrame()
    for pattern_var in pattern_vars:
        pattern_var_df = pd.read_excel(pattern, sheet_name=pattern_var)
        pattern_var_df.set_index('Labels', inplace=True)
        pattern_var_vec = pattern_var_df.values.flatten().tolist()
        # create all combinations of letters and numbers for a 96-well plate
        if plate_size == 96:
            wells = get_well_names(8, 12)
        elif plate_size == 384:
            wells = get_well_names(16, 24)
        else:
            print('Plate size not supported.')
        # create a df with the combinations
        wells_df = pd.DataFrame(wells, columns=['Well'])
        wells_df[pattern_var] = pattern_var_vec
        # concat wells_df to pattern_df
        pattern_df = pd.concat([pattern_df, wells_df], axis=1)
    pattern_vars.append('Well')
    pattern_df = pattern_df[pattern_vars]
    pattern_df['Pattern'] = pattern
    # remove repeated cols
    pattern_df = pattern_df.loc[:,~pattern_df.columns.duplicated()]
    return pattern_df


## MAIN ##
def main():

    # read the design
    design = pd.read_excel(args.input, sheet_name='Design')
    plates = int(args.plates)

    # print the info with colors
    print(f'\nAnalysing the data in {bcolors.OKBLUE}{ROOT}{bcolors.ENDC}, with {bcolors.OKGREEN}{plates}{bcolors.ENDC} plates.\n')

    # create output folder
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)

    # create plots folder
    if not os.path.exists(OUTPUT_PLOTS):
        os.makedirs(OUTPUT_PLOTS)

    # list all the files in the folder
    files = list_files(ROOT)
    

    # get the dates from the files
    dates_full = get_dates(files)

    # get the interval time between each reading
    interval = interval_time(dates_full, plates)
    
    plates_vector = [(i % plates) + 1 for i in range(len(files))]
    files_w_path = [f'{ROOT}/{f}' for f in files]

    print(f'Reading files and creating {bcolors.OKCYAN}dataframes{bcolors.ENDC}: \n')
    final_df = pd.DataFrame()
    with mp.get_context("fork").Pool(8) as p:
        for i, df in enumerate(tqdm(p.imap(hudson_df_reader_parallel, 
                                           zip(files_w_path, dates_full, plates_vector)), 
                                           total=len(files))):
            # concat the dataframes
            final_df = pd.concat([final_df, df], ignore_index=True)
    
    # create a column with Time_h, time in hours, group by plate and well, and fill the column with the time in hours from 0 using the interval 
    final_df['Time_h'] = final_df.groupby(['Plate', 'Well']).cumcount().to_numpy() * interval

    # apply a Wiener function to the OD values per plate and well
    final_df_w = wiener_filter(final_df, interval)
    # change column order in final_df_w to: Plate, Well, Time_h, OD_raw, OD_w
    final_df_w = final_df_w[['Plate', 'Date','Well', 'Time_h', 'OD_raw', 'OD_w']]
    
    ## Get plate size
    plate_size = get_plate_size(final_df_w)

    # if "Pattern" column exists in the design
    if 'Pattern' in design.columns:
        final_pattern_metadata = pd.DataFrame()
        patterns = list(design['Pattern'].unique())
        for pattern in patterns:
            final_pattern_metadata = pd.concat([final_pattern_metadata, get_pattern_metadata(pattern, plate_size)], axis=0)
            final_pattern_metadata.reset_index(drop=True, inplace=True)

        # merge design with final_pattern_metadata
        design = design.merge(final_pattern_metadata, on='Pattern', how='left')

    # calculate AUCs
    print(f'\nCalculating {bcolors.OKCYAN}AUCs{bcolors.ENDC}.\n')
    final_aucs = calculate_aucs(final_df_w) 

    # plot the plates
    print(f'Plotting {bcolors.OKCYAN}plates{bcolors.ENDC}.\n')
    with mp.get_context("fork").Pool(8) as p:
        for _ in tqdm(p.imap(plot_plate_wrapper, zip([final_df_w]*plates, [plates]*plates, [OUTPUT_PLOTS]*plates)), total=len([plates])):
            pass


    # write the dataframes to csv
    final_df_w = design.merge(final_df_w, on=['Plate', 'Well'], how='left')
    final_aucs = design.merge(final_aucs, on=['Plate', 'Well'], how='left')

    final_df_w.to_csv(OUTPUT+'/Timeseries.csv', index=False)
    final_aucs.to_csv(OUTPUT+'/Summary.csv', index=False)
    
if __name__ == '__main__':
    main()