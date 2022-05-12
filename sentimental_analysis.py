from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from .model import TextCnn

# read and convert file contents into Dataframes


def read_and_create_dataframe(file_path):
    """
    Function to read a group of text files
    and convert them into a single dataframe
    """
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


df = read_and_create_dataframe("./dataset/")


# analyze dataframe
print(df["messages"].head(5))

# get the maximum value length
max_value = [len(i) for i in df["messages"]]
max_value_length = max(max_value)

# get the vocabulary_size by using the length of the df
vocabulary_size = len(df)

# perform one hot encoding for text messages
df["encoded"] = df.messages.apply(lambda x: one_hot(x, vocabulary_size))


# add a padding sequences to fill in zero values in encoded vector
X = pad_sequences(df["encoded"], max_value_length, padding="post")

y = df["remarks"]

# split the dataset into two
X_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=30, stratify=y, test_size=0.3)

# get the model
model = TextCnn(vocabulary_size, max_value_length)

# compile the model
