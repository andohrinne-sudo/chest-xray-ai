import h5py

def list_h5_layers(file_path):
    print(f"--- Inspecting Weights: {file_path} ---")
    try:
        with h5py.File(file_path, 'r') as f:
            # In a Keras H5 file, layer names are typically top-level keys
            layer_names = list(f.keys())
            
            # Filter out Keras metadata keys if they exist
            clinical_layers = [name for name in layer_names if name not in ['model_weights', 'optimizer_weights', 'keras_version', 'backend']]
            
            for i, name in enumerate(clinical_layers):
                print(f"Layer {i+1:03d}: {name}")
                
            print(f"--------------------------------------")
            print(f"✅ Total Layers Found: {len(clinical_layers)}")
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    list_h5_layers("pretrained_model.h5")