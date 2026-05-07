import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--thresh', default=0.5)
parser.add_argument('--resolution', default=None)
parser.add_argument('--record', action='store_true')

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Check model path
if not os.path.exists(model_path):
    print('ERROR: Model not found.')
    sys.exit(0)

# Load model
model = YOLO(model_path, task='detect')
labels = model.names

# Determine source type
img_ext_list = ['.jpg','.jpeg','.png','.bmp']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext_list:
        source_type = 'image'
    elif ext.lower() in vid_ext_list:
        source_type = 'video'
    else:
        print('Unsupported file type.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
else:
    print('Invalid source.')
    sys.exit(0)

# Resolution
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

# Recording setup
if record:
    if source_type not in ['video','usb'] or not user_res:
        print('Recording requires video/usb + resolution.')
        sys.exit(0)
    recorder = cv2.VideoWriter('demo1.avi',
                               cv2.VideoWriter_fourcc(*'MJPG'),
                               30, (resW,resH))

# Load source
if source_type == 'image':
    imgs_list = [img_source]

elif source_type == 'folder':
    imgs_list = []
    for file in glob.glob(img_source + '/*'):
        if os.path.splitext(file)[1].lower() in img_ext_list:
            imgs_list.append(file)

elif source_type in ['video','usb']:
    cap = cv2.VideoCapture(img_source if source_type=='video' else usb_idx)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

# Colors
bbox_colors = [(164,120,87),(68,148,228),(93,97,209),(178,182,133),
               (88,159,106),(96,202,231),(159,124,168),(169,162,241),
               (98,118,150),(172,176,184)]

# FPS tracking
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Main loop
while True:
    t_start = time.perf_counter()

    # Load frame
    if source_type in ['image','folder']:
        if img_count >= len(imgs_list):
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1

    else:
        ret, frame = cap.read()
        if not ret:
            break

    # Resize
    if resize:
        frame = cv2.resize(frame,(resW,resH))

    # Run YOLO
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Counters
    object_count = 0
    count_50 = 0
    count_20 = 0

    # Process detections
    for i in range(len(detections)):

        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:

            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label,
                                                  cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)

            cv2.rectangle(frame,
                          (xmin, label_ymin-labelSize[1]-10),
                          (xmin+labelSize[0], label_ymin+baseLine-10),
                          color, cv2.FILLED)

            cv2.putText(frame, label,
                        (xmin, label_ymin-7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,0), 1)

            object_count += 1

            # Count bills
            if classname == "new_50peso_bill":
                count_50 += 1
            elif classname == "new_20peso_bill":
                count_20 += 1

    # Compute total value
    total_value = (count_50 * 50) + (count_20 * 20)

    # Display info
    cv2.putText(frame, f'Objects: {object_count}', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.putText(frame, f'50 Peso Bills: {count_50}', (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.putText(frame, f'20 Peso Bills: {count_20}', (10,90),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.putText(frame, f'Total: PHP {total_value}', (10,120),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)

    # FPS
    if source_type in ['video','usb']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10,150),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.imshow('YOLO detection results', frame)

    if record:
        recorder.write(frame)

    key = cv2.waitKey(0 if source_type in ['image','folder'] else 5)

    if key in [ord('q'), ord('Q')]:
        break

    # FPS calc
    t_stop = time.perf_counter()
    fps = 1/(t_stop - t_start)

    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(fps)

    avg_frame_rate = np.mean(frame_rate_buffer)

# Cleanup
print(f'Average FPS: {avg_frame_rate:.2f}')

if source_type in ['video','usb']:
    cap.release()

if record:
    recorder.release()

cv2.destroyAllWindows()