from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Conv1D, Dropout, GlobalMaxPool1D
import numpy as np
import pandas as pd


# read and convert file contents into Dataframes
def read_and_create_dataframe(file_path):
    import pathlib
    file_path = pathlib.Path(file_path)

    # convert the file_path instance to list
    # search text files in the file_path specified
    file_names = list(file_path.glob("*.txt"))

    # generate the various dataframes of path files
    dataframe_lists = [
        pd.read_csv(filename, sep=".\t", names=[
            "messages", "remarks"], engine="python")
        for filename in file_names
    ]

    # concatenate the various dataframes into one
    df = pd.concat(dataframe_lists)
    return df
