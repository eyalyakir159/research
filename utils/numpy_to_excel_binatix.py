import numpy as np
import pandas as pd
import csv
import json
from datetime import datetime
import h5py
import os

if True:
    with h5py.File('../data/binatix/CCver11a_db_published.mat', 'r') as file:
        print("Keys: %s" % list(file.keys()))

        # Access the 'features' and 'targets' datasets
        if 'features' in file:
            features_data = file['features']
        if 'targets' in file:
            targets_data = file['targets']
        print('converting data to numpy')
        #x = features_data[:]
        y = targets_data[:].squeeze(1)

if False:
    N,S = 300,2
    preds = np.load('../preds.npy')
    trues = np.load('../trues.npy')
    indexstock = np.load('../indexstock.npy')
    data = np.zeros((N,S))

    for t in range(indexstock.shape[0]):
        indexB = indexstock[t][0]
        stockB = indexstock[t][1]
        predB = preds[t]
        truesB = trues[t]
        for i in range(len(indexB)):
            index = int(indexB[i])
            stock = int(stockB[i])-1
            pred = predB[i].item()
            data[index,stock] = pred


data = np.nan_to_num(y[5000:5300],nan=0)+ np.random.randn(300, 3345)*0.3

metadata = {
    "model_name": "Test",
    "create_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    "BB_eval_type": "A",
    "fdb": {
        "train": "CCver11_db_publish.mat",
        "evaluation": "CCver11_db_publish.mat"
    }
}


datacsv = []

# Populate the data list with row number, column number, and the corresponding value
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        datacsv.append([i + 1+5000, j + 1, data[i, j]])

# Write the data to a CSV file
with open('test12345.request.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the JSON metadata as a comment in the first line
    file.write(f'# {json.dumps(metadata)}\n')

    # Write the header
    writer.writerow(['Timestep_idx', 'Instrument_idx', 'Prediction'])

    # Write the data rows from datacsv, not data
    writer.writerows(datacsv)


print("CSV file created successfully.")

