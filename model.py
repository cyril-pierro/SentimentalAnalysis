from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Conv1D, Dropout, GlobalMaxPool1D


def TextCnn(vocabulary_size, max_value_length):
    """
    Text Sentimental Convolutional Network
    """
    embedding_vector_size = 20
    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_vector_size,
              input_length=max_value_length, name='embedding'))
    model.add(Conv1D(64, 3, activation="relu"))
    model.add(GlobalMaxPool1D())
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    model.summary()
