import pandas as pd
import os
from collections import namedtuple

Dataset = namedtuple('Dataset', ['name', 'X', 'y'])

def read_dataset(filename, ext=".txt"):
    # Microarray file
    dataset = pd.read_csv(filename, sep=" ")
    dataset = dataset.transpose()
    X = dataset.drop(0, axis=1).values.astype("float")
    y = dataset[0].values
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]  # Get rid of extension
    return Dataset(name, X, y)
