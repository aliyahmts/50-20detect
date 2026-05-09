import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="Peso Bill Detector", layout="centered")

st.title("🇵🇭 New 50/20 Peso Bill Detector")
st.markdown("Detects new Philippine 50 and 20 Peso bills")

# Load model once
@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model():
    return YOLO('my_model.pt')

model = load_model()
labels = model.names
CONFIDENCE_THRESHOLD = 0.60

def is_valid_bill(class_name):
    return class_name in ["new_50peso_bill", "new_20peso_bill"]

def process_image(image):
    # Convert PIL to OpenCV format (BGR)
    img_array = np.array(image.convert("RGB"))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)   # Important fix
    
    results = model(img_array, verbose=False, conf=CONFIDENCE_THRESHOLD)
    
    count_50 = 0
    count_20 = 0
    annotated = img_array.copy()
    
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
    
    # Convert back to RGB for Streamlit display
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated), count_50, count_20

# Tabs
tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Processing", "Live Camera"])

with tab1:
    st.header("Multiple Image Detection")
    files = st.file_uploader("Upload images", type=["png","jpg","jpeg","bmp"], accept_multiple_files=True)
    
    if files and st.button("Detect Bills", type="primary"):
        total_50 = total_20 = 0
        progress = st.progress(0)
        
        for i, file in enumerate(files):
            image = Image.open(file).convert("RGB")
            annotated, c50, c20 = process_image(image)
            
            total_50 += c50
            total_20 += c20
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_column_width=True)
            with col2:
                st.image(annotated, caption="Detected", use_column_width=True)
            
            progress.progress((i+1)/len(files))
        
        total = total_50*50 + total_20*20
        st.success(f"**Total Value: PHP {total}**")
        st.write(f"50 Peso: {total_50} | 20 Peso: {total_20}")

with tab2:
    st.header("Video Processing")
    video_file = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv"])
    
    if video_file and st.button("Process Video", type="primary"):
        with st.spinner("Processing video..."):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
            count_50 = count_20 = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
                annotated = frame.copy()
                
                for box in results[0].boxes:
                    class_name = labels[int(box.cls.item())]
                    if is_valid_bill(class_name):
                        xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
                        color = (0, 255, 0) if "50" in class_name else (0, 200, 255)
                        cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 3)
                        if class_name == "new_50peso_bill":
                            count_50 += 1
                        else:
                            count_20 += 1
                out.write(annotated)
            
            cap.release()
            out.release()
            
            total = count_50*50 + count_20*20
            st.success("Video processed!")
            st.video(output_path)
            st.write(f"**Total Value:** PHP {total}")
            st.write(f"50₱: {count_50} | 20₱: {count_20}")

with tab3:
    st.header("Live Camera")
    img = st.camera_input("Take a photo of the bills")
    if img:
        image = Image.open(img)
        annotated, c50, c20 = process_image(image)
        st.image(annotated, use_column_width=True)
        total = c50*50 + c20*20
        if total > 0:
            st.success(f"**Detected: PHP {total}**")

st.caption("Built with YOLO • Streamlit Cloud")