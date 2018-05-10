import os

from BSG_Net import BSG_Net
from preprocess import *
from utils import *


def train_model(model):
    pass


if __name__ == '__main__':

    # HYPER PARAMETERS
    # Note: VOCABULARY_SIZE is imported from preprocess.py. It should be set there
    EMBEDDING_DIMENSION = 20
    CONTEXT_SIZE = 3

    # load data:
    data = load_data_from_file('../data/processed/english-french_small/dev.en', CONTEXT_SIZE)
    # Initialize model
    model = BSG_Net(VOCABULARY_SIZE, EMBEDDING_DIMENSION)

    train_model(model)
