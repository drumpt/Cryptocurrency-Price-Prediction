import os
import csv

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
        next(reader) # skip first row
        if sort == "ratio":
            for row in reader: # row : [SNo, Name, Symbol, Date, High, Low, Open, Close, Volume, Marketcap]
                data.append(float(row[7]) / float(row[6]) - 1)

            for i in range(len(data) - look_back - 1):
                input_list.append(data[i : i + look_back])
                output_list.append(data[i + look_back])
        else: # price
            for row in reader:
                input_list.append(float(row[6]))
                output_list.append(float(row[7]))
    return input_list, output_list