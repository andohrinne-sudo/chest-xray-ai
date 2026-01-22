# Chest X-Ray Abnormality Detection

This project originated as a DeepLearning.AI assignment and has **evolved into a production-ready application** for automated chest X-ray analysis, capable of identifying 14 common thoracic abnormalities. It leverages **transfer learning with a DenseNet-121 architecture** to provide reliable and interpretable diagnostic support, demonstrating a robust pipeline for AI in healthcare.

## Model Architecture: DenseNet-121 for Clinical Accuracy

The core of this application is a deep learning model built upon the DenseNet-121 architecture, chosen for its proven performance in medical image analysis and efficient feature reuse.

*   **Backbone Initialization**: The model starts with a `DenseNet121` base, pre-trained on ImageNet (weights loaded from `densenet.hdf5`). The original classification layers (`include_top=False`) are removed to allow for custom classification specific to chest X-ray analysis. This strategic use of a pre-trained backbone accelerates training and improves generalization.
*   **Custom Classification Head**: A custom head is added on top of the DenseNet-121's feature extractor:
    *   A `GlobalAveragePooling2D()` layer reduces the spatial dimensions of the feature maps efficiently.
    *   A `Dense` layer with 14 output units (one for each abnormality) and a `sigmoid` activation function is used. This allows for multi-label classification, where the model predicts the independent probability of presence for each of the 14 abnormalities.
*   **Two-Step Weight Loading Strategy**: 
    1.  The DenseNet-121 backbone is initialized with ImageNet weights.
    2.  Subsequently, the entire model (including the custom head) is loaded with fine-tuned clinical weights from `pretrained_model.h5`. This strategy ensures that the model benefits from both general image features and specific chest X-ray patterns learned during fine-tuning, reflecting **logic evolution** from a generic model to a specialized clinical tool.

## Handling Class Imbalance

Medical datasets, including chest X-rays, often exhibit significant class imbalance, where some abnormalities are much rarer than others. To address this, a custom weighted binary cross-entropy loss function is employed:

*   **Purpose**: The weighted loss function assigns higher penalties to misclassifications of rare classes and lower penalties to common classes, preventing the model from being biased towards prevalent conditions.
*   **Weight Calculation**: The `calculate_loss.py` script computes `pos_weights` and `neg_weights` for each of the 14 abnormalities. These weights are inversely proportional to the prevalence of each pathology in the training dataset. Specifically, `pos_weights` are `1 - prevalence` and `neg_weights` are `prevalence`.
*   **Integration**: These calculated weights are stored in `train_config.json` and are dynamically loaded and utilized by the `XRayModelManager` during model compilation. The `get_weighted_loss` method within `model_manager.py` defines this custom loss function using TensorFlow/Keras backend operations.

## Multi-Label Prediction Logic

The model is designed for multi-label classification, meaning it can predict the presence of multiple abnormalities simultaneously for a single chest X-ray image. The final `Dense` layer, with its `sigmoid` activation, outputs 14 independent probability scores, each corresponding to one of the following abnormalities:

*   Atelectasis
*   Cardiomegaly
*   Effusion
*   Emphysema
*   Fibrosis
*   Hernia
*   Infiltration
*   Mass
*   Nodule
*   Pneumonia
*   Pneumothorax
*   Pleural_Thickening
*   Edema
*   Consolidation

Each score indicates the likelihood of that specific condition being present, independent of others.

## Clinical Evaluation Metrics

Beyond standard accuracy, the model's performance is rigorously evaluated using metrics crucial for clinical applications:

*   **Sensitivity (Recall)**: Measures the proportion of actual positives that are correctly identified as such. High sensitivity is critical for not missing true cases of disease.
*   **Specificity**: Measures the proportion of actual negatives that are correctly identified as such. High specificity helps reduce false alarms.
*   **ROC AUC (Receiver Operating Characteristic Area Under Curve)**: Provides a comprehensive measure of the model's ability to distinguish between positive and negative classes across various threshold settings. A higher ROC AUC indicates better overall diagnostic performance.

These metrics provide a balanced view of the model's effectiveness in a clinical context, addressing both the detection of disease and the avoidance of false positives.

## Preprocessing and Interpretability for Trust

*   **Image Preprocessing**: The `preprocess` method in `model_manager.py` handles image preparation. It scales pixel values to `0-1` and then applies sample-wise zero mean and unit variance normalization, crucial steps for optimal model performance with pre-trained convolutional neural networks.
*   **Model Interpretability with Grad-CAM**: For fostering **trust in AI diagnostics**, the `get_gradcam_layer` method identifies `'conv5_block16_concat'` as the target layer for Grad-CAM visualizations. Grad-CAM generates heatmaps that highlight the specific regions of the X-ray image most influential in the model's prediction, providing visual evidence and aiding clinical decision-making.

## MLOps Proof: Streamlit Application

This project includes a **Streamlit application (`app.py`)** which serves as a proof-of-concept for the model's deployability and integration into a clinical workflow, embodying MLOps principles.

*   **Interactive Interface**: The Streamlit app provides an intuitive web interface for uploading chest X-ray images, running inferences, and visualizing predictions and Grad-CAM heatmaps.
*   **Real-time Inference**: Demonstrates the model's capability to perform real-time diagnostic support.
*   **Easy Deployment**: Showcases how the trained model can be easily packaged and deployed for practical use, moving from development to operational environments seamlessly.

## Setup & Run

### Prerequisites
*   Python 3.8+
*   `pip` package manager

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/andohrinne-sudo/chest-xray-ai.git
    cd chest-xray-ai
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Ensure the necessary model weights and configuration files are present:
    *   `densenet.hdf5` (ImageNet pre-trained weights for DenseNet-121 backbone)
    *   `pretrained_model.h5` (Fine-tuned clinical weights for the chest X-ray task)
    *   `train_config.json` (Generated by `calculate_loss.py`, contains class weights)
    Place `densenet.hdf5` and `pretrained_model.h5` in the root directory of the cloned repository. Run `calculate_loss.py` to generate `train_config.json`.

### Running the Application
To launch the Streamlit web application:
```bash
streamlit run app.py
```
Open your web browser and navigate to the displayed local URL (usually `http://localhost:8501`).
