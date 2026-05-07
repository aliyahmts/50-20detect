from flask import Flask, render_template, request, jsonify
import os
import cv2
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import uuid

# === IMPORTANT: Tell Flask your templates folder is named 'template' ===
app = Flask(__name__, template_folder='template')

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLO model (do this once at startup)
model = YOLO('my_model.pt')   
labels = model.names

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    try:
        # Run inference
        results = model(filepath, verbose=False)
        detections = results[0].boxes

        count_50 = 0
        count_20 = 0

        for box in detections:
            conf = box.conf.item()
            if conf < 0.5:
                continue

            class_id = int(box.cls.item())
            class_name = labels[class_id]

            if class_name == "new_50peso_bill":
                count_50 += 1
            elif class_name == "new_20peso_bill":
                count_20 += 1

        total_value = (count_50 * 50) + (count_20 * 20)

        # Save annotated image
        annotated = results[0].plot()
        annotated_filename = f"annotated_{unique_filename}"
        annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
        cv2.imwrite(annotated_path, annotated)

        return jsonify({
            'success': True,
            'count_50': count_50,
            'count_20': count_20,
            'total_value': total_value,
            'original_image': f'/static/uploads/{unique_filename}',
            'annotated_image': f'/static/uploads/{annotated_filename}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)