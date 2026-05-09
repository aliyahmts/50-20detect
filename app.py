from flask import Flask, render_template, request, jsonify, Response
import os
import cv2
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import uuid
import time

app = Flask(__name__, template_folder='template')

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLO model once at startup
model = YOLO('my_model.pt')
labels = model.names

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

# ===================== IMAGE DETECTION =====================
@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid image'}), 400

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    try:
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
            'type': 'image',
            'count_50': count_50,
            'count_20': count_20,
            'total_value': total_value,
            'original_image': f'/static/uploads/{unique_filename}',
            'annotated_image': f'/static/uploads/{annotated_filename}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===================== VIDEO PROCESSING =====================
@app.route('/detect_video', methods=['POST'])
def detect_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid video'}), 400

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(input_path)

    try:
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_filename = f"processed_{unique_filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        count_50 = 0
        count_20 = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()

            # Count bills (from first frame for simplicity)
            if count_50 == 0 and count_20 == 0:
                for box in results[0].boxes:
                    conf = box.conf.item()
                    if conf >= 0.5:
                        class_name = labels[int(box.cls.item())]
                        if class_name == "new_50peso_bill":
                            count_50 += 1
                        elif class_name == "new_20peso_bill":
                            count_20 += 1

            out.write(annotated_frame)

        cap.release()
        out.release()

        total_value = (count_50 * 50) + (count_20 * 20)

        return jsonify({
            'success': True,
            'type': 'video',
            'count_50': count_50,
            'count_20': count_20,
            'total_value': total_value,
            'processed_video': f'/static/uploads/{output_filename}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===================== LIVE WEBCAM =====================
def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)