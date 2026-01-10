import pandas as pd
import matplotlib.pyplot as plt
from data_loader import XRayDataLoader

# 1. Configuration
LABELS = ["Cardiomegaly", "Emphysema", "Effusion", "Hernia", "Infiltration", "Mass", 
          "Nodule", "Atelectasis", "Pneumothorax", "Pleural_Thickening", "Pneumonia", 
          "Fibrosis", "Edema", "Consolidation"]
METADATA_PATH = 'sample_metadata.csv'

def test_pipeline():
    # Load metadata
    df = pd.read_csv(METADATA_PATH)
    
    # Initialize Loader (batch_size=1 for easy inspection)
    # Note: Ensure you have at least one valid image file in your directory
    loader = XRayDataLoader(df, image_dir='.', labels=LABELS, batch_size=1)
    
    print(f"Total batches in loader: {len(loader)}")
    
    # Fetch the first batch
    try:
        X, y = loader[0]
        print(f"Batch X shape: {X.shape} (Expected: (1, 224, 224, 3))")
        print(f"Batch y shape: {y.shape} (Expected: (1, 14))")
        print(f"Labels for first image: {y[0]}")
        
        # Verify Normalization (DenseNet values should be roughly between -2 and 2)
        print(f"Pixel Range: Min={X.min():.2f}, Max={X.max():.2f}")
        
    except Exception as e:
        print(f"Error during loading: {e}")
        print("Tip: Make sure the 'Image_Path' files exist in your project folder.")

if __name__ == "__main__":
    test_pipeline()
