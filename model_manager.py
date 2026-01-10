import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class XRayModelManager:
    def __init__(self, weights_path="pretrained_model.h5", config_path="train_config.json", labels=None):
        self.labels = labels
        self.weights_path = weights_path
        
        # 1. Load clinical weights from your train_config.json (Stay on track with our plan)
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.pos_weights = np.array(config['pos_weights'])
                self.neg_weights = np.array(config['neg_weights'])
            print(f"✅ Loaded class weights from {config_path}")
        else:
            raise FileNotFoundError(f"Missing {config_path}. Run calculate_loss.py first.")

        # 2. Build the two-step architecture
        self.model = self._build_architecture()

    def get_weighted_loss(self, neg_weights, pos_weights, epsilon=1e-7):
        """Clinical Weighted Loss from the assignment (util.py)."""
        def weighted_loss(y_true, y_pred):
            loss = 0.0
            for i in range(len(neg_weights)):
                loss -= (neg_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon) + 
                         pos_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon))
            return K.sum(loss)
        return weighted_loss

    def _build_architecture(self):
        """
        Two-Step Loading Strategy:
        1. Load backbone weights (densenet.hdf5)
        2. Load fine-tuned clinical weights (pretrained_model.h5)
        """
        # STEP 1: Initialize backbone with densenet.hdf5
        # Note: If this fails, verify the file is in the root chest-xray-ai folder
        base_model = DenseNet121(weights='densenet.hdf5', include_top=False, input_shape=(224, 224, 3))
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(len(self.labels), activation="sigmoid")(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile with our pre-calculated weighted loss
        model.compile(optimizer='adam', 
                      loss=self.get_weighted_loss(self.neg_weights, self.pos_weights))
        
        # STEP 2: Load the fine-tuned weights file
        model.load_weights(self.weights_path)
        print(f"✅ Full clinical model loaded from {self.weights_path}")
        
        return model

    def preprocess(self, img_array):
        """Sample-wise normalization for Zero Mean / Unit Variance."""
        img_array = img_array.astype('float32') / 255.0
        mean = np.mean(img_array)
        std = np.std(img_array)
        normalized_img = (img_array - mean) / (std + 1e-11)
        return np.expand_dims(normalized_img, axis=0)

    def get_gradcam_layer(self):
        return "conv5_block16_concat"