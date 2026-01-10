import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input

class XRayDataLoader(tf.keras.utils.Sequence):
    """
    Standardized Clinical Data Loader for Chest X-ray Multi-label Classification.
    """
    def __init__(self, dataframe, image_dir, labels, batch_size=32, target_size=(224, 224), shuffle=True):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        """Updates indexes after each epoch to prevent overfitting."""
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """Generates one batch of data."""
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_df = self.df.iloc[indexes]
        
        # Load and process batch
        X = np.array([self._load_and_preprocess(path) for path in batch_df['Image_Path']])
        y = batch_df[self.labels].values
        
        return X, y

    def _load_and_preprocess(self, path):
        """
        Loads image and applies DenseNet-specific preprocessing (mean centered).
        """
        img = load_img(f"{self.image_dir}/{path}", target_size=self.target_size)
        img_array = img_to_array(img)
        # Standardize for pre-trained DenseNet weights (vital for clinical accuracy)
        return preprocess_input(img_array)