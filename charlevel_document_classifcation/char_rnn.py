import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Flatten
from keras.layers import Convolution1D
from keras.layers import GlobalMaxPooling1D, MaxPooling1D
from keras.layers import Embedding
from keras.layers import AlphaDropout, Dropout, SpatialDropout1D, BatchNormalization
from keras.layers import ThresholdedReLU
from keras.layers.merge import Add
from keras.callbacks import Callback


NUM_CLASSES = 2
MAX_INPUT_LEN = 1024
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHABET_SIZE = len(ALPHABET)

CHAR_ID = dict()
for idx, char_ in enumerate(ALPHABET):
    CHAR_ID[char_] = idx + 1

def str_to_array(s, input_size=MAX_INPUT_LEN):
    """
    Converting string characters to integer index according to CHAR_ID
    """
    s = s.lower()
    str_index = np.zeros(input_size, dtype='int64')
    max_len = min(len(s), input_size)
    for i in range(1, max_len + 1):
        str_index[i-1] = CHAR_ID.get(s[-i], 0)
    return str_index

def get_data(num_classes=NUM_CLASSES, text_col_name='DOC',
             class_col_name='LABEL',
             input_fname='../data/data.csv'):
    df = pd.read_csv(input_fname)
    one_hot = np.eye(num_classes, dtype='int64')
    X = list()
    y = list()
    for text, class_ in zip(df[text_col_name], df[class_col_name]):
        X.append(str_to_array(text))
        y.append(one_hot[class_])

    return np.asarray(X, dtype='int64'), np.asarray(y)


def get_model(fully_connected_layers,
              hidden_units=[200, 100],
              input_size=MAX_INPUT_LEN,
              embedding_size=500,
              alphabet_size=ALPHABET_SIZE,
              num_classes=2,
              dropout_proba=0.5,
              fixed_embedding=True,
              model='biLSTM',
              loss='categorical_crossentropy'):

    """
    Based on similar ideas/architecture from: https://www.kaggle.com/mamamot/character-based-lstm
    """
    CHAR_ID =
    inputs = Input(shape=(input_size,), name='input_layer', dtype='int64')

    if fixed_embedding:
        embedding_size = alphabet_size
        embedding_weights = np.zeros((alphabet_size + 1, embedding_size))
        for _, idx in CHAR_ID.items():
            onehot = np.zeros(alphabet_size)
            onehot[idx - 1] = 1
            embedding_weights[idx] = onehot

        embeds = Embedding(alphabet_size + 1, embedding_size,
                           input_length=input_size, weights=[embedding_weights], trainable=False)(inputs)
    else:
        embeds = Embedding(alphabet_size + 1, embedding_size, input_length=input_size)(inputs)

    x = embeds
    model_choices = {'biLSTM': lambda hidden_unit: Bidirectional(CuDNNLSTM(hidden_unit, return_sequences=True)),
                     'GRU': lambda hidden_unit: CuDNNGRU(hidden_unit, return_sequences=True)}

    for hidden_unit in hidden_units[:-1]:
        x = model_choices[model](hidden_unit)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_proba)(x)

    model_choices = {'biLSTM': lambda hidden_unit: Bidirectional(CuDNNLSTM(hidden_unit)),
                     'GRU': lambda hidden_unit: CuDNNGRU(hidden_unit)}

    x = model_choices[model](hidden_units[-1])(x)

    x = BatchNormalization()(x)
    x = Dropout(dropout_proba)(x)

    for units in fully_connected_layers:
        x = Dense(units, activation='tanh')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_proba)(x)

    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer='rmsprop', loss=loss, metrics=['accuracy'])
    return model


