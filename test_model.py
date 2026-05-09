from ultralytics import YOLO
import cv2
import os

model = YOLO('my_model.pt')
labels = model.names

print("Model Classes:", labels)
print("=" * 60)

# ================== CHANGE THIS PATH ==================
img_path = r"C:\IS-Project\static\uploads\images (5).jpg"   # ← Put your 500 peso image path here

if not os.path.exists(img_path):
    print(f"Image not found: {img_path}")
    print("Please update the img_path with the correct location of your 500 peso bill image.")
else:
    print(f"Testing image: {img_path}")
    img = cv2.imread(img_path)

    print("\nDetections at different confidence levels:\n")

    for conf_threshold in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        results = model(img, conf=conf_threshold, verbose=False)
        print(f"Threshold = {conf_threshold:.2f} → ", end="")

        detections = results[0].boxes
        if len(detections) == 0:
            print("No detections")
        else:
            for box in detections:
                class_name = labels[int(box.cls.item())]
                conf = box.conf.item()
                print(f"Detected as '{class_name}' with {conf:.4f} confidence")