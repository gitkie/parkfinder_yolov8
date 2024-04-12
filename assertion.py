def process_new_images():
    # Get the list of image files in the directory
    image_files = os.listdir(image_directory)
    
    # Sort the list of image files
    image_files.sort()

    for image_file in image_files:
        # Read the image
        image_path = os.path.join(image_directory, image_file)
        frame = cv2.imread(image_path)

        # Check if the frame is not None and has valid dimensions
        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
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
                                    db.child("carss").push({
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
                    db.child("carss").push({
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

            # Display the frame only if it is valid
            cv2.imshow("RGB", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
