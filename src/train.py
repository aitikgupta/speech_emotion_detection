import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
try: 
  tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
  pass
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, LeakyReLU, Flatten
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop
from keras.initializers import glorot_uniform
from keras.regularizers import l1,l2
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from src.preprocess import generate_features
from src.dataload import load_data
from src.generic_utils import plot_traintest
import keras.utils as np_utils
import numpy as np


def give_model():
    model = Sequential()

    model.add(LSTM(256, recurrent_dropout=0.2, input_shape=(315, 221), kernel_initializer= glorot_uniform(), return_sequences=True))
    model.add(Conv1D(512, 10, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv1D(1024, 10, padding="same"))
    model.add(LeakyReLU())
    model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.2, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(256, recurrent_dropout=0.2))
    model.add(Dense(2048, activation="relu", kernel_regularizer=l1(0.0005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(512, kernel_regularizer=l1(0.0005), kernel_initializer= glorot_uniform(), activation="relu"))
    model.add(Dense(12, activation="softmax"))
    return model

def train_model(dataset=None, model_path="./model/model.h5", labels=None, n_splits=5, learning_rate=0.0001, epochs=30, batch_size=64, verbose=True):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    
    model = give_model()

    if verbose:
        print("[INFO] Training!")
        print(model.summary())

    model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=["acc"])
    
    rlr = ReduceLROnPlateau(factor=0.9, verbose=1, patience=8, monitor="val_loss")
    
    outs = np_utils.to_categorical(labels)
    hist = {}    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset, labels)):
        if verbose:
            print(f"Shape of {fold+1} fold: (Train: {len(train_idx)}, Validation: {len(val_idx)})")
        Xtrain = generate_features(dataset[train_idx], datatype="train")
        ytrain = outs[train_idx]
        Xtest = generate_features(dataset[val_idx], datatype="valid")
        ytest = outs[val_idx]
        if fold !=0:
            model = load_model(model_path)
        h = model.fit(Xtrain, ytrain,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    validation_data=(Xtest, ytest),
                    callbacks=[rlr])
        model.save(model_path)
        hist[fold] = h.history
    if verbose:
        plot_traintest(hist)
