import pandas as pd
import numpy as np
import os
import pickle
from collections import namedtuple
from project import CORPUS_PATH
import logging

Dataset = namedtuple('Dataset', ['name', 'X', 'y'])


def get_corpus_path(corpus_key):
    return CORPUS_PATH[corpus_key]

def describe(dataset):
    print("Dataset: \t\t", dataset.name)
    print("Nro de muestras: \t", dataset.X.shape[0])
    print("Nro de atributos: \t", dataset.X.shape[1])
    classes, counts = np.unique(dataset.y, return_counts=True)
    print("Nro de clases: \t\t", len(classes))
    print("Muestras por clase: ")
    for cls, count in zip(classes, counts):
        print("\t\t\t", cls,"->", count)

def read_dataset(filename, ext=".txt"):
    # Microarray file
    dataset = pd.read_csv(filename, sep=" ")
    dataset = dataset.transpose()
    X = dataset.drop(0, axis=1).values.astype("float")
    y = dataset[0].values
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]  # Get rid of extension
    return Dataset(name, X, y)


def read_datasets(corpus_key, ext=".txt"):
    path = get_corpus_path(corpus_key)

    filenames = [os.path.join(path, f) for f in os.listdir(path)
                 if os.path.isfile(os.path.join(path, f)) and f.endswith(ext)]

    datasets = []
    for filename in filenames:
        if filename.endswith(ext):
            datasets.append(read_dataset(filename))

    return datasets


def save_pickle(filename, data):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, "wb") as file:
        pickle.dump(data, file)


def load_pickle(filename):
    try:
        with open(filename, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        logging.exception("Archivo no encontrado.")
        return None

# Deprecated
# def dataset_generator(path):
#     filenames = glob.glob(path)
#     for filename in filenames:
#         if filename.endswith(".mat"):
#             # Matlab file
#             mat = scipy.io.loadmat(filename)
#             X = mat['X']
#             X = X.astype(float)
#             y = mat['Y']
#             y = y[:, 0]
#             yield Dataset(filename, X, y)
#
#         if filename.endswith(".txt"):
#             # Microarray file
#             dataset = pd.read_csv(filename, sep=" ")
#             dataset = dataset.transpose()
#             X = dataset.drop(0, axis=1).values
#             y = dataset[0].values
#             yield Dataset(filename, X, y)
