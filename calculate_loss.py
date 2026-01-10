import pandas as pd
import numpy as np
import json

# 1. Load your newly uploaded training data
train_df = pd.read_csv("data/nih_new/train-small.csv")

LABELS = ["Cardiomegaly", "Emphysema", "Effusion", "Hernia", "Infiltration", 
          "Mass", "Nodule", "Atelectasis", "Pneumothorax", "Pleural_Thickening", 
          "Pneumonia", "Fibrosis", "Edema", "Consolidation"]

def calculate_clinical_weights(df, labels):
    pos_weights = []
    neg_weights = []
    
    for label in labels:
        # Calculate prevalence (P) of each pathology
        prevalence = np.mean(df[label])
        
        # Calculate weights based on the specific math for this model
        pos_weights.append(1 - prevalence)
        neg_weights.append(prevalence)
        
    return pos_weights, neg_weights

pos, neg = calculate_clinical_weights(train_df, LABELS)

# 2. Save to your config file for the ModelManager
with open('train_config.json', 'w') as f:
    json.dump({"pos_weights": pos, "neg_weights": neg}, f)

print("✅ 'train_config.json' has been updated with weights from your new dataset.")