import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import mysql.connector

# Initialize YOLO model
model = YOLO("yolov8s.pt")

# Connect to MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="db_acas",
)

# Create a cursor object
cursor = conn.cursor()


# Mouse event callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

# RTSP stream URL
rtsp_url = "rtsp://parkfinder:capstone2@192.168.1.5:554/stream2"

# Attempt to open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the RTSP stream was opened successfully
if not cap.isOpened():
    print(f"Error: Failed to open RTSP stream '{rtsp_url}'.")
    exit()

# RTSP stream was successfully opened
print(f"RTSP stream '{rtsp_url}' opened successfully.")

# Read class names from coco.txt file
with open("coco.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

# Fetch parking areas from the database in ascending order
cursor.execute("SELECT area FROM Carss ORDER BY CAST(SUBSTRING(area, 5) AS UNSIGNED)")
rows = cursor.fetchall()
areas = [row[0] for row in rows]

# Define parking area polygons
area_points_dict = {
    "AREA1": [(159, 358), (252, 368), (264, 174), (170, 170)],
    "AREA2": [(267, 364), (356, 369), (364, 170), (278, 168)],
    "AREA3": [(375, 369), (470, 371), (472, 170), (379, 168)],
    "AREA4": [(487, 372), (584, 374), (582, 171), (489, 168)],
    "AREA5": [(602, 372), (691, 371), (685, 175), (600, 170)],
}

# Convert keys to uppercase
area_points_dict = {key.upper(): value for key, value in area_points_dict.items()}


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
                    for area_name, area_points in zip(areas, area_points_dict):
                        results_area = cv2.pointPolygonTest(
                            np.array(
                                area_points_dict[area_name.replace(" ", "").upper()],
                                np.int32,
                            ),
                            (cx, cy),
                            False,
                        )
                        if results_area >= 0:
                            cv2.rectangle(
                                frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (0, 255, 0),
                                2,
                            )
                            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                            cv2.putText(
                                frame,
                                str(c),
                                (int(x1), int(y1)),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )
                            # Insert car information into the database
                            cursor.execute(
                                """
                                INSERT INTO Carss (area, x, y, date, time, status)
                                VALUES (%s, %s, %s, CURDATE(), CURTIME(), 'occupied')
                                ON DUPLICATE KEY UPDATE x=VALUES(x), y=VALUES(y), date=VALUES(date), time=VALUES(time), status='occupied'
                            """,
                                (area_name, cx, cy),
                            )
                            conn.commit()
                            break

    # Display parking area numbers and available spaces
    for i, (area_name, area_points) in enumerate(
        zip(areas, area_points_dict.values()), start=1
    ):
        car_count = sum(
            1
            for result in results
            for index, row in pd.DataFrame(result.boxes.xyxy).iterrows()
            if not row.empty
            and cv2.pointPolygonTest(
                np.array(area_points, np.int32),
                ((int(row[0]) + int(row[2])) // 2, (int(row[1]) + int(row[3])) // 2),
                False,
            )
            >= 0
        )
        if car_count == 0:
            # If no cars are detected, mark the parking area as vacant
            cursor.execute(
                """
                INSERT INTO Carss (area, x, y, date, time, status)
                VALUES (%s, NULL, NULL, CURDATE(), CURTIME(), 'vacant')
                ON DUPLICATE KEY UPDATE x=NULL, y=NULL, date=VALUES(date), time=VALUES(time), status='vacant'
                """,
                (area_name,),
            )
            conn.commit()

        if car_count == 1:
            cv2.polylines(
                frame, [np.array(area_points, np.int32)], True, (0, 0, 255), 2
            )
        else:
            cv2.polylines(
                frame, [np.array(area_points, np.int32)], True, (0, 255, 0), 2
            )
        cv2.putText(
            frame,
            str(i),
            (area_points[0][0], area_points[0][1] - 20),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        cv2.putText(
            frame,
            str(car_count),
            (area_points[0][0], area_points[0][1]),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Close MySQL connection
cursor.close()
conn.close()
cv2.destroyAllWindows()