import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import firebase_admin
from firebase_admin import credentials, db

# Initialize YOLO model
model = YOLO("yolov8s.pt")

# Firebase credentials
cred = credentials.Certificate("path/to/firebase_credentials.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'YOUR_DATABASE_URL'
})

# Get a reference to the Firebase Realtime Database
ref = db.reference('/')

# Directory where images are saved
image_directory = "screenshots"

# Define the file extension for images
image_extension = ".jpg"

# Read class names from coco.txt file
with open("coco.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

# Define parking area polygons
area_points_dict = {
    "AREA1": [(76, 666), (214, 692), (225, 422), (84, 414)],
    "AREA2": [(231, 691), (396, 704), (405, 423), (242, 418)],
    "AREA3": [(413, 705), (592, 709), (595, 426), (419, 422)],
    "AREA4": [(611, 711), (791, 712), (788, 429), (613, 427)],
    "AREA5": [(807, 707), (969, 701), (964, 431), (803, 429)],
    "AREA6": [(984, 694), (1123, 678), (1114, 430), (978, 430)],

    "AREA7": [(151, 56), (280, 47), (243, 261), (105, 264)],
    "AREA8": [(290, 46), (435, 43), (413, 259), (258, 262)],
    "AREA9": [(454, 42), (601, 42), (598, 263), (429, 254)],
    "AREA10": [(614, 42), (767, 50), (782, 269), (615, 264)],
    "AREA11": [(784, 52), (919, 63), (950, 273), (797, 266)],
    "AREA12": [(934, 71), (1060, 85), (1099, 276), (965, 276)]
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
                        for area_name, area_points in area_points_dict.items():
                            results_area = cv2.pointPolygonTest(
                                np.array(area_points, np.int32),
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
                                # Insert car information into the Firebase Realtime Database
                                ref.child("carss").push({
                                    "area": area_name,
                                    "x": cx,
                                    "y": cy,
                                    "date": time.strftime("%Y-%m-%d"),
                                    "time": time.strftime("%H:%M:%S"),
                                    "status": "occupied"
                                })
                                break

        # Display parking area numbers and available spaces
        for i, (area_name, area_points) in enumerate(
            area_points_dict.items(), start=1
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
                ref.child("carss").push({
                    "area": area_name,
                    "x": None,
                    "y": None,
                    "date": time.strftime("%Y-%m-%d"),
                    "time": time.strftime("%H:%M:%S"),
                    "status": "vacant"
                })

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
    time.sleep(1)  # Adjust this value as needed

# Close MySQL connection
# cursor.close()
# conn.close()
cv2.destroyAllWindows()
