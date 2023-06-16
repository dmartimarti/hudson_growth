## Documentation for the Hudson machine reader

This script is meant to be used with the data collected by the Hudson reader in the Cabreiro lab. 

The way this works is slightly different from the other script to analyse Biospa data. As the output from the Hudson reader are several csv files, one per time and read, you'll need to supply *how many plates* are you analysing as well as where they files are located. For example, if you are analysing 4 different plates, the reader will first read plate 1, then plate 2 and so on. 

About the Design.xlsx and Pattern files, it works more or less similar. Instead of having to write the file, you only need to include the plate number. For example, if you are analysing 4 plates, there should be also 4 plates as rows in the Design file. Each one of them can have a Pattern file that has the same structure as with the other code, that will ease the analysis.

The output of this code will be two csv files: one with the AUCs (calculated as the trapezoid of the data), and another with the Timeseries. There will be a separate folder with some very simple plots of the data per plate. 
There is another difference with the output files, as now the Timeseries file will be ready to be used in R, no need of further transformation post-hoc. 

Finally, the code uses some tiny bits of parallelisation, so you can specify how many threads you want to use. It should make things a bit faster. 

### Requirements

You can create a new conda environment for this. The packages needed are: pandas, numpy, matplotlib, tqdm and openpyxl. You can install them by running:

```
conda install pandas numpy matplotlib tqdm openpyxl
```

### Usage

```
python hudson_reader.py -i <input Design file (Default is "Design.xlsx")> \
                        -o <output folder> \
                        -f <folder where the csv files are located>
                        -p <number of plates tested>
                        -t <threads to use (Default is 1)>
```

For example, for the test data included in the repo, you should run
    
    ```
    python hudson_reader.py -i Design.xlsx -f ./test_data -p 4 -o Output -t 4
    ```

