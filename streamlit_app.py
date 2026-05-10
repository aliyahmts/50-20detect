import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import tempfile

st.set_page_config(page_title="New 50/20 Peso Bill Detector", layout="centered")

# Inject local CSS from static/style.css
with open("static/style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("50/20 Peso Bill Detector")

# ========================= MODEL LOADING =========================
@st.cache_resource(show_spinner="Loading YOLO Model...")
def load_model():
    try:
        model_path = 'my_model.pt'
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found in the app directory!")
            st.stop()
        model = YOLO(model_path)
        st.success("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

model = load_model()
labels = model.names
CONFIDENCE_THRESHOLD = 0.65

def is_valid_bill(class_name):
    return class_name in ["new_50peso_bill", "new_20peso_bill"]

# ========================= FRAME PROCESSING =========================
def process_frame(frame):
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    annotated = frame.copy()
    
    print(f"Detected {len(results[0].boxes)} objects")

    for box in results[0].boxes:
        class_name = labels[int(box.cls.item())]
        conf = box.conf.item()
        if is_valid_bill(class_name) and conf >= CONFIDENCE_THRESHOLD:
            xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
            color = (0, 255, 0) if "50" in class_name else (0, 200, 255)
            
            cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 3)
            label = f"{class_name.replace('new_', '').replace('_peso_bill', ' Peso')}"
            cv2.putText(annotated, label, (xyxy[0], xyxy[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated

# ========================= TABS =========================
tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Processing", "Live Webcam"])

# ======================== IMAGE TAB ========================
with tab1:
    st.subheader("Multiple Image Detection")
    
    uploaded_files = st.file_uploader(
        "Upload one or more images", 
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.markdown("### Selected Images Preview")
        cols = st.columns(min(4, len(uploaded_files)))
        
        for i, file in enumerate(uploaded_files):
            with cols[i % len(cols)]:
                image = Image.open(file)
                st.image(image, caption=file.name, use_column_width=True)
        
        if st.button("🚀 Detect All Images", type="primary", use_container_width=True):
            total_50 = total_20 = 0
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                try:
                    image = Image.open(file).convert("RGB")
                    img_array = np.array(image)
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    annotated_cv = process_frame(img_cv)
                    annotated = Image.fromarray(cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB))
                    
                    # Count bills
                    results = model(img_cv, verbose=False, conf=CONFIDENCE_THRESHOLD)
                    for box in results[0].boxes:
                        class_name = labels[int(box.cls.item())]
                        if is_valid_bill(class_name):
                            if class_name == "new_50peso_bill":
                                total_50 += 1
                            else:
                                total_20 += 1
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption=f"Original: {file.name}")
                    with col2:
                        st.image(annotated, caption=f"Detected: {file.name}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
            
            total_value = (total_50 * 50) + (total_20 * 20)
            st.success("Detection Complete!")
            st.metric("**Total Value**", f"PHP {total_value:,}")
            st.write(f"**50 Peso Bills:** {total_50}  **20 Peso Bills:** {total_20}")

# ======================== VIDEO TAB ========================
with tab2:
    st.subheader("Video Processing")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="video_uploader")
    
    if uploaded_video:
        st.video(uploaded_video)
        
        if st.button("Process Video", type="primary"):
            with st.spinner("Processing video... This may take a while"):
                try:
                    # Save uploaded video
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_video.getbuffer())
                    tfile.close()

                    cap = cv2.VideoCapture(tfile.name)
                    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

                    total_50 = total_20 = 0
                    frame_count = 0
                    FRAME_SKIP = 6  # Change to 3-8 depending on video length/speed

                    progress_bar = st.progress(0)

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        
                        # Process every frame for smooth annotated video
                        annotated = process_frame(frame)
                        out.write(annotated)
                        
                        # Count on selected frames only
                        if frame_count % FRAME_SKIP == 0:
                            results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
                            for box in results[0].boxes:
                                class_name = labels[int(box.cls.item())]
                                if is_valid_bill(class_name):
                                    if class_name == "new_50peso_bill":
                                        total_50 += 1
                                    else:
                                        total_20 += 1
                        
                        if total_frames > 0:
                            progress_bar.progress(min(1.0, frame_count / total_frames))

                    cap.release()
                    out.release()

                    # Simple deduplication (each bill likely appears in many frames)
                    total_50 = max(0, total_50 // (FRAME_SKIP * 2))
                    total_20 = max(0, total_20 // (FRAME_SKIP * 2))

                    total_value = (total_50 * 50) + (total_20 * 20)

                    st.success("Video processed successfully!")
                    st.metric("**Total Value**", f"PHP {total_value:,}")
                    st.write(f"**50 Peso Bills:** {total_50}  **20 Peso Bills:** {total_20}")

                    # Show output video
                    with open(output_path, "rb") as f:
                        st.video(f.read())

                    # Cleanup temp files
                    for path in [tfile.name, output_path]:
                        try:
                            os.unlink(path)
                        except:
                            pass

                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
# ======================== WEBCAM TAB ========================
with tab3:
    st.subheader("Live Webcam")
    st.info("Take a clear photo using your camera")
    
    img = st.camera_input("Capture Image")
    if img:
        try:
            image = Image.open(img)
            img_array = np.array(image.convert("RGB"))
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            annotated_cv = process_frame(img_cv)
            annotated = Image.fromarray(cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB))
            
            st.image(annotated, caption="Detection Result", use_column_width=True)
            
            # Count
            results = model(img_cv, verbose=False, conf=CONFIDENCE_THRESHOLD)
            count_50 = count_20 = 0
            for box in results[0].boxes:
                class_name = labels[int(box.cls.item())]
                if is_valid_bill(class_name):
                    if class_name == "new_50peso_bill":
                        count_50 += 1
                    else:
                        count_20 += 1
            
            total = count_50 * 50 + count_20 * 20
            if total > 0:
                st.success(f"**Detected Value: PHP {total}**")
            else:
                st.warning("No valid 50 or 20 peso bills detected")
        except Exception as e:
            st.error(f"Error: {e}")