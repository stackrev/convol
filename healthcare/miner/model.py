# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 demon

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import tempfile
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import bittensor as bt
from constant import Constant

class CustomModelCheckpoint(Callback):
    def __init__(self, model, path, save_freq, monitor='val_loss'):
        super(CustomModelCheckpoint, self).__init__()
        self.model = model
        self.path = path
        self.save_freq = save_freq
        self.monitor = monitor
        self.best = np.Inf
        self.batch_counter = 0

    def on_batch_end(self, batch, logs=None):
        self.batch_counter += 1
        current = logs['loss']
        if self.best == np.Inf:
            self.best = current
        if self.batch_counter % self.save_freq == 0:
            if current is not None and current < self.best:
                bt.logging.info(f"\nBest Model saved!!! {self.best}, {current}")
                self.best = current
                self.model.save(os.path.join(self.path, 'best_model'.format(self.batch_counter)))

class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def load_and_preprocess_image(self, image_path, target_size=(224, 224)):
        try:
            # Load image
            img = load_img(image_path, target_size=target_size)
            img_array = img_to_array(img)

            # Resize the image using NumPy's resize. Note: np.resize and PIL's resize behave differently.
            img_array = np.array(image.smart_resize(img_array, target_size))

            # Normalize the image
            img_array = img_array / 255.0

            # Expand dimensions to fit the model input format
            # img_array = np.expand_dims(img_array, axis=0)

            return img_array
        except Exception as e:
            return "ERROR"

    def generate_data(self, image_paths, labels, batch_size):
        num_samples = len(image_paths)
        while True:
            for offset in range(0, num_samples, batch_size):
                batch_images = []
                batch_labels = labels[offset:offset+batch_size]
                for img_path in image_paths[offset:offset+batch_size]:
                    absolute_path = Constant.BASE_DIR + '/healthcare/dataset/miner/images/' + img_path
                    img = self.load_and_preprocess_image(absolute_path)
                    if isinstance(img, str):
                        continue
                    batch_images.append(img)
                yield np.array(batch_images), np.array(batch_labels)

    # Function to check if an image exists (mock implementation)
    def image_exists(self, image_name, target_size=(224, 224)):
        image_path = Constant.BASE_DIR + '/healthcare/dataset/miner/images/' + image_name
        if not os.path.exists(image_path):
            return False
        return True

    def load_dataframe(self):
        # Load CSV file
        dataframe = pd.read_csv(Constant.BASE_DIR + '/healthcare/dataset/miner/Data_Entry.csv')

        # Preprocess image names and labels of dataframe
        # String list and corresponding image list
        string_list = dataframe['label']
        image_list = dataframe['image_name']

        # Split the strings into individual labels
        split_labels = [set(string.split('|')) for string in string_list]

        # Initialize MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        binary_array_full = mlb.fit_transform(split_labels)

        # Filter out rows where the corresponding image does not exist
        binary_array_filtered = [binary_array_full[i] for i, image in enumerate(image_list) if self.image_exists(image)]
        
        if not binary_array_filtered:
            bt.logging.error("No images found")
            return False, False, False

        binary_array_filtered = np.vstack(binary_array_filtered)
        
        # Filter out rows where the file does not exist
        dataframe['file_exists'] = dataframe['image_name'].apply(lambda x: self.image_exists(x))
        dataframe = dataframe[dataframe['file_exists']]
        dataframe = dataframe.drop(columns=['file_exists'])
        
        image_paths = dataframe['image_name'].values

        train_gen = self.generate_data(image_paths, binary_array_filtered, self.config.batch_size)

        num_classes = binary_array_filtered.shape[1]

        return train_gen, dataframe, num_classes

    def get_model(self, num_classes):
        model_file_path = Constant.BASE_DIR + '/healthcare/models/best_model'
        
        # Check if model exists
        if not self.config.restart and os.path.exists(model_file_path):
            model = load_model(model_file_path)
            bt.logging.info(f"Model loaded")
            return model
        
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),
            # Add more layers as needed
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='sigmoid')  # num_classes based on your dataset
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def train(self):
        train_generator, train_df, num_classes = self.load_dataframe()

        if train_generator == False:
            return
            
        model = self.get_model(num_classes)

        checkpoint = ModelCheckpoint(
            filepath=Constant.BASE_DIR + '/healthcare/models/best_model', 
            monitor='loss', 
            verbose=1, 
            save_best_only=True, 
            mode='auto'
        )

        custom_checkpoint = CustomModelCheckpoint(
            model,
            path=Constant.BASE_DIR + '/healthcare/models/',
            save_freq=self.config.save_model_period  # Change this to your preferred frequency
        )

        # Add EarlyStopping
        early_stopping = EarlyStopping(monitor='loss', patience=10)

        if self.config.num_epochs == -1:
            while True:
                history = model.fit(
                    train_generator,
                    steps_per_epoch=len(train_df) // self.config.batch_size,  # Adjust based on your batch size
                    epochs=1,  # Number of epochs
                    callbacks=[checkpoint, early_stopping]
                )
        else:
            history = model.fit(
                train_generator,
                steps_per_epoch=len(train_df) // self.config.batch_size,  # Adjust based on your batch size
                epochs=self.config.num_epochs,  # Number of epochs
                callbacks=[checkpoint, early_stopping]
            )