import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv5n model
# model = cv2.dnn.readNet('yolov5n.pt', 'yolov5n.yaml')
model = YOLO('yolov5n.pt', 'yolov5n.yaml')

# Define class names
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Define parking area polygons
areas = {
    "area1": [(159, 358), (252, 368), (264, 174), (170, 170)],
    "area2": [(267, 364), (356, 369), (364, 170), (278, 168)],
    "area3": [(375, 369), (470, 371), (472, 170), (379, 168)],
    "area4": [(487, 372), (584, 374), (582, 171), (489, 168)],
    "area5": [(602, 372), (691, 371), (685, 175), (600, 170)]
}

# Initialize video capture from RTSP stream
rtsp_url = 'rtsp://parkfinder:capstone2@192.168.1.3:554/stream2'
cap = cv2.VideoCapture(rtsp_url)

# Check if the RTSP stream was opened successfully
if not cap.isOpened():
    print(f"Error: Failed to open RTSP stream '{rtsp_url}'.")
    exit()
print(f"RTSP stream '{rtsp_url}' opened successfully.")


while True:
    ret, frame = cap.read()
    if not ret:
        break
    time.sleep(1)
    frame = cv2.resize(frame, (800, 500))

    # Perform object detection using YOLOv5n model
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    layer_names = model.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    outs = model.forward(output_layers)

    # Iterate over detected objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Minimum confidence threshold
                # Extract bounding box coordinates
                center_x = int(detection[0] * 800)
                center_y = int(detection[1] * 500)
                w = int(detection[2] * 800)
                h = int(detection[3] * 500)
                # Calculate top-left corner coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
                cv2.putText(frame, class_list[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Check if object centroid is inside any parking area polygon
                for area_name, area_points in areas.items():
                    results_area = cv2.pointPolygonTest(np.array(area_points, np.int32), (center_x, center_y), False)
                    if results_area >= 0:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
                        cv2.putText(frame, class_list[class_id], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1)
                        break

    # Display parking area numbers and available spaces
    for i, (area_name, area_points) in enumerate(areas.items(), start=1):
        car_count = sum(1 for out in outs for detection in out if detection[4] >= 0.5 and
                        cv2.pointPolygonTest(np.array(area_points, np.int32),
                                             ((int(detection[0] * 800) + int(detection[2] * 800)) // 2,
                                              (int(detection[1] * 500) + int(detection[3] * 500)) // 2),
                                             False) >= 0)
        if car_count == 1:
            cv2.polylines(frame, [np.array(area_points, np.int32)], True, (0, 0, 255), 2)
        else:
            cv2.polylines(frame, [np.array(area_points, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str(i), (area_points[0][0], area_points[0][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1)
        cv2.putText(frame, str(car_count), (area_points[0][0], area_points[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
