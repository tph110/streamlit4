"""
app.py

Streamlit app for 8-class skin lesion classification including SCC detection.
Model automatically downloads from HuggingFace on first run.

Deployment:
    streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import requests
from pathlib import Path

try:
    import timm
except ImportError:
    st.error("timm not installed. Please add 'timm' to requirements.txt")
    st.stop()

# -------------------------
# Configuration
# -------------------------
MODEL_URL = "https://huggingface.co/Skindoc/streamlit4/resolve/main/best_model_20251115_205238.pth"
MODEL_PATH = "best_model_20251115_205238.pth"
MODEL_NAME = "tf_efficientnet_b4"
NUM_CLASSES = 8
IMG_SIZE = 384

# Class names (alphabetical order - matches training)
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'scc', 'vasc']

# Clinical descriptions
CLASS_DESCRIPTIONS = {
    'akiec': 'Actinic Keratoses / Intraepithelial Carcinoma',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis-like Lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus',
    'scc': 'Squamous Cell Carcinoma',
    'vasc': 'Vascular Lesion'
}

# Risk categories
RISK_INFO = {
    'akiec': {'level': 'Moderate Risk', 'type': 'Pre-cancerous', 'color': '🟡', 'action': 'Dermatologist evaluation recommended'},
    'bcc': {'level': 'High Risk', 'type': 'Malignant', 'color': '🔴', 'action': 'Immediate dermatologist referral required'},
    'bkl': {'level': 'Low Risk', 'type': 'Benign', 'color': '🟢', 'action': 'Routine monitoring recommended'},
    'df': {'level': 'Low Risk', 'type': 'Benign', 'color': '🟢', 'action': 'Routine monitoring recommended'},
    'mel': {'level': 'High Risk', 'type': 'Malignant', 'color': '🔴', 'action': 'Urgent dermatologist referral required'},
    'nv': {'level': 'Low Risk', 'type': 'Benign', 'color': '🟢', 'action': 'Routine monitoring recommended'},
    'scc': {'level': 'High Risk', 'type': 'Malignant', 'color': '🔴', 'action': 'Immediate dermatologist referral required'},
    'vasc': {'level': 'Low Risk', 'type': 'Benign', 'color': '🟢', 'action': 'Routine monitoring recommended'}
}

# -------------------------
# Model Download and Loading
# -------------------------
@st.cache_resource
def download_model():
    """Download model from HuggingFace if not already present"""
    if not os.path.exists(MODEL_PATH):
        st.info("⏳ Downloading model from HuggingFace (this may take a few minutes on first run)...")
        
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(MODEL_PATH, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        downloaded += len(chunk)
                        f.write(chunk)
                        
            st.success("✓ Model downloaded successfully!")
            
        except Exception as e:
            st.error(f"Error downloading model: {e}")
            st.error("Please check your internet connection and try again.")
            st.stop()
    
    return MODEL_PATH

@st.cache_resource
def load_model():
    """Load the trained model"""
    # Download model if needed
    model_path = download_model()
    
    try:
        # Create model architecture
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        return model
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# -------------------------
# Image Preprocessing
# -------------------------
def get_transform():
    """Get image preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.05)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# -------------------------
# Prediction Functions
# -------------------------
def predict_single(model, image_tensor):
    """Make a single prediction"""
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0))
        probabilities = F.softmax(outputs, dim=1)
    
    probs = probabilities.cpu().numpy()[0]
    return probs

def predict_with_tta(model, image, n_augmentations=5):
    """
    Make prediction using Test-Time Augmentation.
    Uses horizontal flip, vertical flip, and rotations.
    """
    transform_base = get_transform()
    
    # Define augmentations
    augmentations = [
        lambda x: x,  # Original
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),  # Horizontal flip
        lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),   # Vertical flip
        lambda x: x.rotate(90),   # Rotate 90
        lambda x: x.rotate(270),  # Rotate 270
    ]
    
    all_probs = []
    
    with torch.no_grad():
        for aug_fn in augmentations[:n_augmentations]:
            # Apply augmentation and transform
            aug_image = aug_fn(image.copy())
            image_tensor = transform_base(aug_image)
            
            outputs = model(image_tensor.unsqueeze(0))
            probabilities = F.softmax(outputs, dim=1)
            all_probs.append(probabilities.cpu().numpy()[0])
    
    # Average probabilities across augmentations
    mean_probs = np.mean(all_probs, axis=0)
    std_probs = np.std(all_probs, axis=0)
    
    return mean_probs, std_probs

# -------------------------
# Streamlit UI
# -------------------------
def main():
    # Page config
    st.set_page_config(
        page_title="Skin Lesion Classifier",
        page_icon="🔬",
        layout="wide"
    )
    
    # Title and description
    st.title("🔬 AI-Powered Skin Lesion Classification")
    st.markdown("""
    This AI model classifies dermoscopic images into 8 categories including **Squamous Cell Carcinoma (SCC)**.
    
    **Model Performance:**
    - Overall F1-Score: **0.84**
    - SCC F1-Score: **0.80**
    - Trained on 25,000+ dermoscopic images (ISIC2019 dataset)
    """)
    
    st.warning("⚠️ **Medical Disclaimer:** This is a screening tool only. Always consult a qualified dermatologist for diagnosis.")
    
    # Load model
    with st.spinner("Loading AI model..."):
        model = load_model()
    
    # Sidebar
    st.sidebar.header("Settings")
    use_tta = st.sidebar.checkbox("Use Test-Time Augmentation (TTA)", value=True, 
                                    help="TTA improves accuracy but takes longer")
    
    if use_tta:
        n_tta = st.sidebar.slider("Number of TTA augmentations", 3, 5, 5)
    
    show_all_probs = st.sidebar.checkbox("Show all class probabilities", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About the Classes")
    st.sidebar.markdown("""
    - **AKIEC**: Actinic Keratoses (Pre-cancerous)
    - **BCC**: Basal Cell Carcinoma (Malignant)
    - **BKL**: Benign Keratosis (Benign)
    - **DF**: Dermatofibroma (Benign)
    - **MEL**: Melanoma (Malignant)
    - **NV**: Melanocytic Nevus (Benign)
    - **SCC**: Squamous Cell Carcinoma (Malignant)
    - **VASC**: Vascular Lesion (Benign)
    """)
    
    # File uploader
    st.header("📤 Upload Dermoscopic Image")
    uploaded_file = st.file_uploader(
        "Choose a dermoscopic image (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear dermoscopic image of the skin lesion"
    )
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Input Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)
            st.caption(f"Image size: {image.size[0]} × {image.size[1]} pixels")
        
        with col2:
            st.subheader("🤖 AI Analysis")
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                if use_tta:
                    mean_probs, std_probs = predict_with_tta(model, image, n_tta)
                else:
                    transform = get_transform()
                    image_tensor = transform(image)
                    mean_probs = predict_single(model, image_tensor)
                    std_probs = np.zeros_like(mean_probs)
            
            # Get prediction
            predicted_idx = np.argmax(mean_probs)
            predicted_class = CLASS_NAMES[predicted_idx]
            confidence = float(mean_probs[predicted_idx])
            
            # Get risk info
            risk_info = RISK_INFO[predicted_class]
            
            # Display results
            st.markdown(f"### {risk_info['color']} Predicted Diagnosis")
            st.markdown(f"## **{predicted_class.upper()}**")
            st.markdown(f"*{CLASS_DESCRIPTIONS[predicted_class]}*")
            
            # Confidence
            st.metric("Confidence", f"{confidence:.1%}")
            
            if use_tta:
                uncertainty = float(std_probs[predicted_idx])
                st.metric("Uncertainty (±)", f"{uncertainty:.3f}")
                
                if uncertainty < 0.05:
                    st.success("✓ High confidence prediction")
                elif uncertainty < 0.15:
                    st.warning("⚠ Moderate confidence prediction")
                else:
                    st.error("⚠⚠ Low confidence - expert review recommended")
            
            # Risk level
            st.markdown("---")
            st.markdown(f"### {risk_info['color']} Risk Assessment")
            st.info(f"**Level:** {risk_info['level']} ({risk_info['type']})")
            st.info(f"**Recommendation:** {risk_info['action']}")
        
        # Probability distribution
        if show_all_probs:
            st.markdown("---")
            st.subheader("📊 Probability Distribution")
            
            # Sort by probability
            sorted_indices = np.argsort(mean_probs)[::-1]
            
            for idx in sorted_indices:
                class_name = CLASS_NAMES[idx]
                prob = mean_probs[idx]
                risk_info = RISK_INFO[class_name]
                
                # Create columns for better layout
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    st.markdown(f"{risk_info['color']} **{class_name.upper()}**")
                
                with col2:
                    st.progress(float(prob))
                
                with col3:
                    st.markdown(f"**{prob:.1%}**")
        
        # Clinical recommendations
        st.markdown("---")
        st.subheader("⚕️ Clinical Recommendations")
        
        if predicted_class in ['mel', 'bcc', 'scc']:
            st.error("""
            **⚠ URGENT: Malignant lesion detected**
            
            - Immediate dermatologist referral strongly recommended
            - Biopsy advised for confirmation
            - Early treatment is critical for best outcomes
            """)
            
            if predicted_class == 'mel':
                st.info("📌 Melanoma requires prompt surgical excision")
            elif predicted_class == 'scc':
                st.info("📌 SCC treatment may include excision, radiation, or topical therapy")
            elif predicted_class == 'bcc':
                st.info("📌 BCC typically has excellent prognosis with treatment")
        
        elif predicted_class == 'akiec':
            st.warning("""
            **⚠ Pre-cancerous lesion detected**
            
            - Dermatologist evaluation recommended
            - Treatment may prevent progression to cancer
            - Options include cryotherapy, topical agents, or photodynamic therapy
            """)
        
        else:
            st.success("""
            **✓ Benign lesion detected**
            
            - Routine monitoring recommended
            - Consult dermatologist if lesion changes
            - Annual skin check advised
            """)
        
        # Download results
        st.markdown("---")
        if st.button("📥 Download Detailed Report"):
            report = f"""
SKIN LESION CLASSIFICATION REPORT
================================

PREDICTED DIAGNOSIS: {predicted_class.upper()}
Description: {CLASS_DESCRIPTIONS[predicted_class]}

CONFIDENCE: {confidence:.1%}
Risk Level: {risk_info['level']} ({risk_info['type']})

PROBABILITY DISTRIBUTION:
"""
            for idx in sorted_indices:
                class_name = CLASS_NAMES[idx]
                prob = mean_probs[idx]
                report += f"\n{class_name.upper():8s}: {prob:.1%}"
            
            report += f"""

RECOMMENDATION:
{risk_info['action']}

IMPORTANT DISCLAIMER:
This is an AI screening tool, NOT a medical diagnosis.
Always consult a qualified dermatologist for proper diagnosis.
Any concerning lesion warrants professional evaluation.

Model Information:
- Architecture: EfficientNet-B4
- Training Dataset: ISIC2019 (25,000+ images)
- Overall F1-Score: 0.84
- SCC F1-Score: 0.80
"""
            
            st.download_button(
                label="Download Report as TXT",
                data=report,
                file_name=f"skin_lesion_report_{predicted_class}.txt",
                mime="text/plain"
            )
    
    else:
        st.info("👆 Please upload a dermoscopic image to begin analysis")
        
        # Example image placeholder
        st.markdown("---")
        st.subheader("📸 Image Requirements")
        st.markdown("""
        For best results, please ensure:
        - Image is a **dermoscopic (dermatoscope) photo**
        - Lesion is **clearly visible** and **in focus**
        - Good **lighting** without shadows
        - Image shows the **entire lesion**
        - Acceptable formats: JPG, JPEG, PNG
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>Developed using the ISIC2019 dataset</strong></p>
        <p>Model automatically downloads from <a href='https://huggingface.co/Skindoc/streamlit4'>HuggingFace</a></p>
        <p style='color: #666; font-size: 0.9em;'>
            This tool is for research and educational purposes only.<br>
            Not intended to replace professional medical advice, diagnosis, or treatment.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
