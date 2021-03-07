import os
import csv

import torch

def get_all_csv_files(data_dir):
    csv_files = []
    for file in os.listdir(data_dir):
        if os.path.splitext(file)[1] == ".csv":
            csv_files.append(os.path.join(data_dir, file))
    return csv_files

def read_csv(csv_dir):
    input_list, output_list = [], []
    with open(csv_dir, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader) # skip first row
        for row in reader: # row : [SNo, Name, Symbol, Date, High, Low, Open, Close, Volume, Marketcap]
            input_list.append(float(row[7]) / float(row[6]))
        input_tensor = torch.tensor(input_list[:-1], dtype = torch.float)
        output_tensor = torch.tensor(input_list[1:], dtype = torch.float)
    return input_tensor, output_tensor