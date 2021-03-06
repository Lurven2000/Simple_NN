import pandas as pd
import numpy as np

def get_data_sets():
    examples = [
        np.array([
            [0, 0],
            [1, 1]
        ]),
        np.array([
            [1, 1],
            [0, 0]
        ]),
        np.array([
            [1, 0],
            [0, 0]
        ]),
        np.array([
            [0, 1],
            [0, 0]
        ]),
        np.array([
            [0, 0],
            [1, 0]
        ]),
        np.array([
            [0, 0],
            [0, 1]
        ]),
        np.array([
            [1, 0],
            [1, 0]
        ]),
        np.array([
            [0, 1],
            [0, 1]
        ]),
        np.array([
            [1, 0],
            [1, 1]
        ]),
        np.array([
            [0, 1],
            [1, 1]
        ]),
        np.array([
            [1, 1],
            [0, 1]
        ]),
        np.array([
            [1, 1],
            [1, 0]
        ]),
    ]

    def training_set():
        while True:
            index = np.random.choice(len(examples))
            yield examples[index]

    def test_set():
        while True:
            index = np.random.choice(len(examples))
            yield examples[index]


    return training_set, test_set