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


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_predict = [x[1] for x in val_predict]
        val_targ = self.validation_data[1]
        val_targ = [x[1] for x in val_targ]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print ("— val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
        return

metrics = Metrics()

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


# Fixed embedding, from https://github.com/BrambleXu/nlp-beginner-guide-keras/blob/master/char-level-cnn/char_cnn.py

# Embedding weights
embedding_weights = []  # (70, 69)
embedding_weights.append(np.zeros(vocab_size))  # (0, 69)

for char, i in tk.word_index.items():  # from index 1 to 69
    onehot = np.zeros(vocab_size)
    onehot[i - 1] = 1
    embedding_weights.append(onehot)

embedding_weights = np.array(embedding_weights)
print('Load')

# Embedding layer Initialization
embedding_layer = Embedding(vocab_size + 1,
                            embedding_size,
                            input_length=input_size,
                            weights=[embedding_weights])


def define_model_1(conv_layers,
                  fully_connected_layers,
                  input_size=MAX_INPUT_LEN,
                  embedding_size=32,
                  alphabet_size=ALPHABET_SIZE,
                  num_classes=2, optimizer='adam',
                  dropout_proba=0.5, fl_activation='selu',
                  fl_initializer='lecun_normal',
                  conv_activations='tanh',
                  loss='categorical_crossentropy'):
    """
    Based on: https://arxiv.org/abs/1508.06615
    """
    inputs = Input(shape=(input_size,), name='input_layer', dtype='int64')
    embeds = Embedding(alphabet_size + 1, embedding_size, input_length=input_size)(inputs)
    convs = list()
    for num_filters, filter_width in conv_layers:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=filter_width,
                             activation=conv_activations,
                             name='ConvLayer{}{}'.format(num_filters, filter_width))(embeds)
        pool = GlobalMaxPooling1D(name='MaxPoolLayer{}{}'.format(num_filters, filter_width))(conv)
        convs.append(pool)

    x = Concatenate()(convs)
    for units in fully_connected_layers:
        x = Dense(units, activation=fl_activation, kernel_initializer=fl_initializer)(x)
        x = AlphaDropout(dropout_proba)(x)

    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


def define_model_2(conv_layers,
                   fully_connected_layers,
                   threshold = 1e-6,
                   num_classes=2,
                   optimizer='adam',
                   dropout_proba=0.5,
                   alphabet_size=ALPHABET_SIZE,
                   embedding_size=32,
                   input_size=MAX_INPUT_LEN,
                   loss='categorical_crossentropy'):

    """
    Based on https://arxiv.org/abs/1509.01626
    """
    inputs = Input(shape=(input_size,), name='input_layer', dtype='int64')
    x = Embedding(alphabet_size + 1, embedding_size, input_length=input_size)(inputs)
    for num_filters, filter_width, max_pool in conv_layers:
        x = Convolution1D(filters=num_filters,
                          kernel_size=filter_width)(x)
        x = ThresholdedReLU(threshold)(x)
        if max_pool != -1:
            x = MaxPooling1D(max_pool)(x)

    x = Flatten()(x)
    for units in fully_connected_layers:
        x = Dense(units)(x)
        x = ThresholdedReLU(threshold)(x)
        x = Dropout(dropout_proba)(x)

    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def define_model_3(conv_layers, fully_connected_layers,
                   dropout_proba=0.5,
                   embedding_size=32,
                   alphabet_size=ALPHABET_SIZE,
                   num_classes=2, optimizer='adam',
                   loss='categorical_crossentropy'):
    """
    Based on: https://arxiv.org/abs/1803.01271
    """
    inputs = Input(shape=(input_size,), name='input', dtype='int64')
    x = Embedding(alphabet_size + 1, embedding_size, input_length=input_size)(inputs)

    def _residual_block(num_filters, filter_width, dilation, dropout_proba, x):
        x = Convolution1D(num_filters, filter_width, padding='same', dilation_rate=dilation, activation='linear')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout1D(dropout_proba)(x)
        return x

    for num_filters, filter_width in conv_layers:
        res_block_in = Convolution1D(num_filters, filter_width, padding='same', activation='linear')(x)
        x = _residual_block(num_filters, filter_width, 1, dropout_proba, x)
        x = _residual_block(num_filters, filter_width, 2, dropout_proba, x)
        x = Add()([res_block_in, x])

    x = Flatten()(x)

    for units in fully_connected_layers:
        x = Dense(units)(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_proba)(x)

    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizer, loss=loss)
    return model


X, y = get_data()

print(X.shape)

TRAIN_LEN = int(X.shape[0] * 0.8)

# Define model then train
# model = ...
# model.fit(X[:TRAIN_LEN], y[:TRAIN_LEN],
#         validation_data=(X[TRAIN_LEN:], y[TRAIN_LEN:]),
#          epochs=300,
#          batch_size=128,
#          callbacks=[metrics])

# https://www.kaggle.com/kmader/character-level-cnn-classification-with-dilations


file_path="best_weights.h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

callbacks_list = [checkpoint, early] #early
model.fit(X_t_train, y_train,
          validation_data=(X_t_test, y_test),
          batch_size=batch_size,
          epochs=epochs,
          shuffle = True,
          callbacks=callbacks_list)


