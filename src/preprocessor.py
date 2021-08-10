import os
import csv

import torch
from sklearn.preprocessing import MinMaxScaler

def get_all_csv_files(data_dir):
    csv_files = []
    for file in os.listdir(data_dir):
        if os.path.splitext(file)[1] == ".csv":
            csv_files.append(os.path.join(data_dir, file))
    return csv_files

def read_csv(csv_dir, look_back, sort):
    data, input_list, output_list = [], [], []
    with open(csv_dir, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader) # skip first header row

        for row in reader:
            data.append(float(row[6]))
        data.append(float(row[7]))

        if sort == "ratio":
            data = [price / data[0] - 1 for price in data] # normalization
            for i in range(len(data) - look_back - 1):
                input_list.append(data[i : i + look_back])
                output_list.append(data[i + look_back])
        else: # price
            input_list.extend(data[:-1])
            output_list.extend(data[1:])
    return input_list, output_list