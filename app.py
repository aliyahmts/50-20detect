from flask import Flask, render_template, request, jsonify, Response
import os
import cv2
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__, template_folder='template')

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = YOLO('my_model.pt')
labels = model.names

CONFIDENCE_THRESHOLD = 0.83

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


def is_valid_bill(class_name):
    return class_name in ["new_50peso_bill", "new_20peso_bill"]


# ===================== MULTIPLE IMAGE DETECTION =====================
@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400

    total_50 = 0
    total_20 = 0
    annotated_urls = []

    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            continue

        # Save file
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(filepath)

        # Detect
        img = cv2.imread(filepath)
        results = model(img, verbose=False, conf=CONFIDENCE_THRESHOLD)

        count_50 = 0
        count_20 = 0
        filtered_boxes = []

        for box in results[0].boxes:
            class_name = labels[int(box.cls.item())]
            conf = box.conf.item()
            if is_valid_bill(class_name) and conf >= CONFIDENCE_THRESHOLD:
                filtered_boxes.append(box)
                if class_name == "new_50peso_bill":
                    count_50 += 1
                else:
                    count_20 += 1

        total_50 += count_50
        total_20 += count_20

        # Annotate image
        annotated = img.copy()
        for box in filtered_boxes:
            xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
            color = (0, 255, 0) if "50" in labels[int(box.cls.item())] else (0, 200, 255)
            cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 3)
            label = f"{labels[int(box.cls.item())].replace('new_', '').replace('_peso_bill', ' Peso')}: {box.conf.item():.2f}"
            cv2.putText(annotated, label, (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        ann_filename = f"annotated_{unique_name}"
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], ann_filename), annotated)
        annotated_urls.append(f'/static/uploads/{ann_filename}')

    total_value = (total_50 * 50) + (total_20 * 20)

    return jsonify({
        'success': True,
        'count_50': total_50,
        'count_20': total_20,
        'total_value': total_value,
        'images': annotated_urls
    })

# ===================== FRAME PROCESSOR =====================
def process_frame(frame):
    results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
    annotated = frame.copy()
    
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
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_filename = f"processed_{unique_filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        count_50 = 0
        count_20 = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = process_frame(frame)
            out.write(annotated_frame)

            if count_50 == 0 and count_20 == 0:
                for box in model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)[0].boxes:
                    class_name = labels[int(box.cls.item())]
                    if is_valid_bill(class_name):
                        if class_name == "new_50peso_bill":
                            count_50 += 1
                        else:
                            count_20 += 1

        cap.release()
        out.release()

        total_value = (count_50 * 50) + (count_20 * 20)

        return jsonify({
            'success': True,
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
        annotated_frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)