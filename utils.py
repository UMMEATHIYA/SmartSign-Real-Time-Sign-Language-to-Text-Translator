import os
import numpy as np
import pandas as pd

def load_data(data_path='data/sequences/', sequence_length=30):
    X, y = [], []
    label_map = {}
    label_idx = 0

    for file in os.listdir(data_path):
        if not file.endswith('.csv'):
            continue
        label = file.split('_')[0]

        if label not in label_map:
            label_map[label] = label_idx
            label_idx += 1

        df = pd.read_csv(os.path.join(data_path, file))
        if df.shape[0] == sequence_length:
            X.append(df.values)
            y.append(label_map[label])

    X = np.array(X)
    y = np.array(y)

    return X, y, label_map
