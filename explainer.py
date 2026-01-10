import tensorflow as tf
import numpy as np
import cv2

class XRayExplainer:
    """
    Expert-level Explainability module using Grad-CAM for medical imaging.
    """
    def __init__(self, model_manager):
        self.model = model_manager.model
        # Automatically find the last convolutional layer (usually 'relu' in DenseNet)
        self.last_conv_layer_name = model_manager.get_gradcam_layer()

    def generate_heatmap(self, processed_img, class_index):
        # Construct a model that outputs the final CONV layer and the prediction
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            # Forward pass to get the feature maps and logits
            conv_outputs, predictions = grad_model(processed_img)
            loss = predictions[:, class_index]

        # Calculate gradients of the loss w.r.t. the last conv layer output
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of the gradients (importance weights)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weighted sum of the feature maps
        heatmap = conv_outputs[0] @ weights[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU and normalization for clinical visualization
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()