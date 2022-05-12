from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Conv1D, Dropout, GlobalMaxPool1D


def TextCnn():
    """
    Text Sentimental Convolutional Network
    """
    embedding_vector_size = 20
