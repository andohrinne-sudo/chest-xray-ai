import streamlit as st
import numpy as np
import cv2
from PIL import Image
from model_manager import XRayModelManager
from explainer import XRayExplainer

# --- 1. CLINICAL SETUP ---
LABELS = [
    'Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 
    'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening', 
    'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation'
]

HEALTHY_MEAN_THRESHOLD = 1e-3
HEALTHY_STD_THRESHOLD = 1e-3

# --- 2. RESOURCE LOADING ---
@st.cache_resource
def load_diagnostic_system():
    manager = XRayModelManager(weights_path="pretrained_model.h5", config_path="train_config.json", labels=LABELS)
    explainer = XRayExplainer(manager)
    return manager, explainer

manager, explainer = load_diagnostic_system()

# --- 3. UI INFRASTRUCTURE ---
st.set_page_config(page_title="Clinical X-Ray AI", layout="wide")
st.title("🩺 Chest X-Ray Clinical Diagnostic Aid")
st.warning("**Research Tool Only:** Not for primary diagnosis.")

uploaded_file = st.file_uploader("Upload X-Ray", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # --- 4. IMAGE ACQUISITION & PREPROCESSING ---
    img = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(img.resize((224, 224)))
    
    # Process image BEFORE sidebar calls it
    processed_img = manager.preprocess(img_array)
    
    # --- 5. REGULATED SIDEBAR SYSTEM HEALTH ---
    st.sidebar.header("System Health")
    mean_val = np.mean(processed_img)
    std_val = np.std(processed_img)
    
    is_mean_healthy = abs(mean_val) < HEALTHY_MEAN_THRESHOLD
    is_std_healthy = abs(std_val - 1.0) < HEALTHY_STD_THRESHOLD

    st.sidebar.metric("Mean Intensity", f"{mean_val:.4f}", delta="0.0000", delta_color="off")
    st.sidebar.metric("Signal Variance", f"{std_val:.4f}", delta="1.0000", delta_color="off")

    if not (is_mean_healthy and is_std_healthy):
        st.sidebar.error("⚠️ Pipeline Uncalibrated")
        if st.sidebar.button("🔄 Calibration Reset"):
            st.rerun() 
    else:
        st.sidebar.success("✅ System Calibrated")

    # --- 6. MAIN ANALYSIS UI ---
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Scan", use_container_width=True)
    
    with col2:
        st.subheader("Clinical Analysis")
        with st.spinner("Analyzing anatomy..."):
            preds = manager.model.predict(processed_img)[0]
        
        sorted_indices = np.argsort(preds)[::-1]
        for idx in sorted_indices:
            if preds[idx] > 0.5:
                st.error(f"**{LABELS[idx]}**: {preds[idx]:.2%}")
            elif preds[idx] > 0.1:
                st.warning(f"{LABELS[idx]}: {preds[idx]:.2%}")

    # --- 7. EXPLAINABILITY ---
    st.divider()
    target_disease = st.selectbox("Explain findings for:", LABELS, index=int(sorted_indices[0]))
    heatmap = explainer.generate_heatmap(processed_img, LABELS.index(target_disease))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (224, 224))), cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), 0.4, img_array, 0.6, 0)
    st.image(overlayed, caption=f"Anatomical Focus: {target_disease}", width=600)

else:
    st.info("Awaiting patient scan for analysis. Upload an image to activate system health monitoring.")