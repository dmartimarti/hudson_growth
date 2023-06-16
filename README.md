## Documentation for the Hudson machine reader


dependencies to install 
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

