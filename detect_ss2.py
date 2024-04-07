import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import mysql.connector
import os

# Initialize YOLO model
model = YOLO("yolov8s.pt")

# Connect to MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="capstone2",
    database="db_acas",
)

# Create a cursor object
cursor = conn.cursor()

# Directory where images are saved
image_directory = "screenshots"

# Define the file extension for images
image_extension = ".jpg"

# Read class names from coco.txt file
with open("coco.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

# Fetch parking areas from the database in ascending order
cursor.execute("SELECT area FROM carss ORDER BY CAST(SUBSTRING(area, 5) AS UNSIGNED)")
rows = cursor.fetchall()
areas = [row[0] for row in rows]

# Define parking area polygons
area_points_dict = {
    "AREA1": [(249, 517), (401, 523), (429, 260), (283, 260)],
    "AREA2": [(429, 521), (574, 526), (580, 258), (449, 261)],
    "AREA3": [(605, 523), (754, 526), (741, 257), (603, 261)],
    "AREA4": [(772, 527), (918, 520), (894, 264), (757, 260)],
    "AREA5": [(947, 523), (1073, 513), (1032, 264), (915, 264)],
}

# Convert keys to uppercase
area_points_dict = {key.upper(): value for key, value in area_points_dict.items()}

# Function to process new images
def process_new_images():
    # Get the list of image files in the directory
    image_files = os.listdir(image_directory)
    
    # Sort the list of image files
    image_files.sort()

    for image_file in image_files:
        # Read the image
        image_path = os.path.join(image_directory, image_file)
        frame = cv2.imread(image_path)

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
                                    INSERT INTO carss (area, x, y, date, time, status)
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
                    INSERT INTO carss (area, x, y, date, time, status)
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

# Main loop to continuously process new images
while True:
    process_new_images()
    time.sleep(3)  # Adjust this value as needed

# Close MySQL connection
cursor.close()
conn.close()
cv2.destroyAllWindows()
