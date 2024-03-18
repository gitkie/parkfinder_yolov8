import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

# Initialize YOLO model
model = YOLO('yolov8s.pt')
# model = YOLO('yolov5n.pt', 'yolov5n.yaml')

# Mouse event callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Video capture
#cap = cv2.VideoCapture('parking1.mp4')


#RTSP===============================================================
rtsp_url = 'rtsp://parkfinder:capstone2@192.168.1.3:554/stream2'

# Attempt to open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the RTSP stream was opened successfully
if not cap.isOpened():
    print(f"Error: Failed to open RTSP stream '{rtsp_url}'.")
    exit()

# RTSP stream was successfully opened
print(f"RTSP stream '{rtsp_url}' opened successfully.")

#RTSP===============================================================


# Read class names from coco.txt file
with open("coco.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

# Define parking area polygons
areas = {
    "area1": [(159, 358), (252, 368), (264, 174), (170, 170)],
    "area2": [(267, 364), (356, 369), (364, 170), (278, 168)],
    "area3": [(375, 369), (470, 371), (472, 170), (379, 168)],
    "area4": [(487, 372), (584, 374), (582, 171), (489, 168)],
    "area5": [(602, 372), (691, 371), (685, 175), (600, 170)],
    #"area6": [(336, 343), (357, 410), (409, 408), (382, 340)],
    #"area7": [(396, 338), (426, 404), (479, 399), (439, 334)],
    #"area8": [(458, 333), (494, 397), (543, 390), (495, 330)],
    #"area9": [(165, 290), (226, 290), (235, 178), (174, 174)],
    #"area10": [(564, 323), (615, 381), (654, 372), (596, 315)],
    #"area11": [(616, 316), (666, 369), (703, 363), (642, 312)],
    #"area12": [(674, 311), (730, 360), (764, 355), (707, 308)]
}

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    time.sleep(1)
    frame = cv2.resize(frame, (800, 500))

    # Perform object detection using YOLO model
    results = model.predict(frame)

    # Iterate over detected objects
    for result in results:
        # Extract bounding box coordinates
        px = pd.DataFrame(result.boxes.xyxy)

        # Iterate over each detected object's bounding box
        if not px.empty:
            for index, row in px.iterrows():
                if len(row) >= 4:  # Check if row contains all expected values
                    x1, y1, x2, y2 = row[:4]
                    c = class_list[0]  # Placeholder for class name
                    cx = int((x1 + x2) // 2)
                    cy = int((y1 + y2) // 2)

                    # Check if car centroid is inside any parking area polygon
                    for area_name, area_points in areas.items():
                        results_area = cv2.pointPolygonTest(np.array(area_points, np.int32), (cx, cy), False)
                        if results_area >= 0:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                            cv2.putText(frame, str(c), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                            break

    # Display parking area numbers and available spaces
    for i, (area_name, area_points) in enumerate(areas.items(), start=1):
        car_count = sum(1 for result in results for index, row in pd.DataFrame(result.boxes.xyxy).iterrows()
                        if not row.empty and cv2.pointPolygonTest(np.array(area_points, np.int32),
                                                                   ((int(row[0]) + int(row[2])) // 2,
                                                                    (int(row[1]) + int(row[3])) // 2),
                                                                   False) >= 0)
        if car_count == 1:
            cv2.polylines(frame, [np.array(area_points, np.int32)], True, (0, 0, 255), 2)
        else:
            cv2.polylines(frame, [np.array(area_points, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, str(i), (area_points[0][0], area_points[0][1] - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (0, 0, 0), 1)
        cv2.putText(frame, str(car_count), (area_points[0][0], area_points[0][1]), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (255, 255, 255), 1)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
