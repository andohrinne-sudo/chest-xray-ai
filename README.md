🩺 Clinical Chest X-Ray AI Diagnostic
An AI-driven diagnostic tool utilizing DenseNet-121 for 14-label pathology detection.

🚀 Features
Clinical Preprocessing: Sample-wise normalization (Zero Mean/Unit Variance).

Interpretability: Grad-CAM anatomical localization.

Architecture: Two-step weight loading (Backbone + Fine-tuned).

🛠 Setup
pip install -r requirements.txt

Ensure densenet.hdf5 and pretrained_model.h5 are in the root.

streamlit run app.py