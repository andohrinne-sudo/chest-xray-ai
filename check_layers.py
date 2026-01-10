import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet121

# 1. Load the model (Ensure you use the same architecture as app.py)
model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))

# 2. Iterate in reverse to find the final 4D layer
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        # Convolutional layers typically output 4D tensors (Batch, Height, Width, Filters)
        if len(layer.output.shape) == 4 and "conv" in layer.name.lower():
            return layer.name
    return None

last_layer_name = find_last_conv_layer(model)
print(f"✅ The detected last convolutional layer is: {last_layer_name}")