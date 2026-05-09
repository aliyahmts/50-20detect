import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="New 50/20 Peso Bill Detector", layout="centered")

# Custom styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E3A8A; text-align: center; margin-bottom: 10px;}
    .stButton>button {width: 100%; background-color: #2563EB; color: white; height: 50px; font-size: 1.1rem;}
</style>
""", unsafe_allow_html=True)

st.title("New 50/20 Peso Bill Detector")

# Load Model
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return YOLO('my_model.pt')

model = load_model()
labels = model.names
CONFIDENCE_THRESHOLD = 0.65   # Lowered for better detection

def is_valid_bill(class_name):
    return class_name in ["new_50peso_bill", "new_20peso_bill"]

def process_image(image):
    img_array = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    results = model(img_cv, verbose=False, conf=CONFIDENCE_THRESHOLD)
    annotated = img_cv.copy()
    
    count_50 = count_20 = 0
    
    for box in results[0].boxes:
        class_name = labels[int(box.cls.item())]
        conf = box.conf.item()
        
        if is_valid_bill(class_name) and conf >= CONFIDENCE_THRESHOLD:
            xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
            color = (0, 255, 0) if "50" in class_name else (0, 200, 255)
            
            cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 3)
            label = f"{class_name.replace('new_', '').replace('_peso_bill', ' Peso')}: {conf:.2f}"
            cv2.putText(annotated, label, (xyxy[0], xyxy[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if class_name == "new_50peso_bill":
                count_50 += 1
            else:
                count_20 += 1
                
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), count_50, count_20

# Tabs
tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Processing", "Live Webcam"])

with tab1:
    st.subheader("Multiple Image Detection")
    
    uploaded_files = st.file_uploader(
        "Upload images", 
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
        help="You can select multiple images"
    )
    
    # Preview Section
    if uploaded_files:
        st.markdown("### Selected Images Preview")
        cols = st.columns(min(3, len(uploaded_files)))  # Show up to 3 per row
        
        for i, file in enumerate(uploaded_files):
            with cols[i % len(cols)]:
                image = Image.open(file)
                st.image(image, caption=file.name, use_column_width=True)
        
        # Detect Button
        if st.button("🚀 Detect Bills", type="primary", use_container_width=True):
            total_50 = total_20 = 0
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                image = Image.open(file).convert("RGB")
                annotated, c50, c20 = process_image(image)
                
                total_50 += c50
                total_20 += c20
                
                # Show results
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption=f"Original: {file.name}", use_column_width=True)
                with col2:
                    st.image(annotated, caption=f"Detected: {file.name}", use_column_width=True)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            total_value = (total_50 * 50) + (total_20 * 20)
            
            st.success("Detection Complete!")
            st.metric("Total Value", f"PHP {total_value:,}")
            st.write(f"**50 Peso Bills:** {total_50} | **20 Peso Bills:** {total_20}")

with tab3:
    st.subheader("Live Webcam")
    img = st.camera_input("Take a clear photo of the bills")
    if img:
        image = Image.open(img)
        annotated, c50, c20 = process_image(image)
        st.image(annotated, use_column_width=True)
        total = c50*50 + c20*20
        if total > 0:
            st.success(f"**Detected Value: PHP {total}**")

st.caption("Built with YOLO • Streamlit Cloud")