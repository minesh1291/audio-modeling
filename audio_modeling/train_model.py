import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.applications as apps
from tensorflow.keras.utils import Sequence

import librosa

from sklearn.model_selection import train_test_split 


# Funtion Definitions
#--------------------
def get_base_direcroty(path: str) -> str:
    return path.split("/")[-1]

def get_file_extention(filename: str) -> str:
    return filename.split(".")[-1]

def assign_class_to_files(data_dir, white_list=["npy"]):
    out_list = []
    for direcroty, _, files in os.walk(data_dir):
        for filename in files:
            if get_file_extention(filename) not in white_list:
                continue
            path = os.path.join(direcroty, filename)
            class_name = get_base_direcroty(direcroty)
            out_list.append((path, class_name))
    return out_list
    
def calc_spectogram(arr: np.array):
    w_size = 1024
    hop_len = 110
    stft = librosa.core.stft(arr, n_fft=w_size, hop_length=hop_len)
    stft_spect = np.abs(stft)
    stft_spect_log = librosa.amplitude_to_db(stft_spect)
    return stft_spect_log

def calc_features(arr: np.array):
    feat_arr = np.resize(calc_spectogram(arr), (513, 513))
    return feat_arr


class AudioSequence(Sequence):
    def __init__(self, x_set: np.array, y_set: np.array, batch_size: int):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        print(self.batch_size, self.x[:3])
        
    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))
    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        x2d = np.array([
                np.apply_along_axis(calc_features, 0, np.load(file_name))    
                 for file_name in batch_x])
        
        x3d = np.repeat(np.expand_dims(x2d, axis=3), 3, axis=3)
        
        return x3d, np.array(batch_y)
    
    
def get_CNN_model():
    # input layer
    new_input = layers.Input(shape=(513, 513, 3))

    model = keras.models.Sequential([
        new_input,
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(1024, activation="relu"),
        # layers.Dropout(0.5),
        # layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(264, activation="softmax")
    ])
    
    # compile model
    # adam = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    # summarize
    print(model.summary())
    print("\n\n Defined model.")
    return model

def get_dense_model():
    # input layer
    new_input = layers.Input(shape=(513, 513, 3))

    model = keras.models.Sequential([
        new_input,
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        # layers.Dropout(0.5),
        # layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(264, activation="softmax")
    ])
    
    # compile model
    # adam = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    # summarize
    print(model.summary())
    print("\n\n Defined model.")
    return model


def get_model():
    # input layer
    new_input = layers.Input(shape=(513, 513, 3))

    
    # model = apps.VGG16(
    model = apps.InceptionResNetV2(
       include_top=False, 
       input_tensor=new_input, 
       weights="imagenet", 
    )

    # freeze layer weights
    for layer in model.layers:
        layer.trainable = False

    # add new classifier layers
    flat1 = layers.Flatten()(model.output)
    # class1 = layers.Dense(1024, activation='relu')(flat1)
    # class1 = layers.Dense(512, activation='relu')(flat1)
    class1 = layers.Dense(312, activation='relu')(flat1)
    output = layers.Dense(264, activation='softmax')(class1)

    # define new model
    model = keras.models.Model(inputs=model.inputs, outputs=output)

    # compile model
    # adam = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    # summarize
    # model.summary()
    print("\n\n Defined model.")
    return model


def main():
    
    TRAIN_DIR = "./train_dir/"
    N_EPOCHS = 5
    BATCH_SIZE =  64 # //2 # 264  # *5 # 32, 64, 128, 264
    SEED = 1291
    
    np.random.seed(SEED)
    idx_train, idx_test = train_test_split(range(227510), test_size=0.2, random_state=1)
    idx_train, idx_val = train_test_split(idx_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
    print(f"\n\n {len(idx_train)} {len(idx_val)} {len(idx_test)}")


    sample_list = assign_class_to_files(TRAIN_DIR)
    samples_df = pd.DataFrame(sample_list, columns=["path", "class"])
    # samples_df.shape

    
    train_gen = AudioSequence(x_set=samples_df["path"].values[idx_train], y_set=keras.utils.to_categorical(pd.Categorical(samples_df["class"]).codes)[idx_train], batch_size=BATCH_SIZE)
    val_gen = AudioSequence(x_set=samples_df["path"].values[idx_val], y_set=keras.utils.to_categorical(pd.Categorical(samples_df["class"]).codes)[idx_val], batch_size=BATCH_SIZE)
    test_gen = AudioSequence(x_set=samples_df["path"].values[idx_test], y_set=keras.utils.to_categorical(pd.Categorical(samples_df["class"]).codes)[idx_test], batch_size=BATCH_SIZE)
    
    
    # model = get_model()
    # model = get_dense_model()
    model = get_CNN_model()
    
    
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, mode="min", restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath="best_model.ckp", save_best_only=True)
    ]
    
    
    print("\n\n Training using data iterator")
    model.fit(train_gen, epochs=5, batch_size=BATCH_SIZE,  verbose=1, callbacks=callbacks, validation_data=val_gen) 
    # validation_data=val_gen, callbacks=callbacks, )
    
    config = {
        "BATCH_SIZE": BATCH_SIZE,
        "N_EPOCHS": N_EPOCHS,
        # "": 
    }
    print(f"\n\n config: {config}")

if __name__ == "__main__":
    main()
