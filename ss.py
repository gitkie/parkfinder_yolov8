import cv2
import time
import os

# RTSP stream URL
rtsp_url = "rtsp://username:password@your_ip_address:port/stream"

# Output directory to save screenshots
output_directory = "screenshots"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

while True:
    # Capture RTSP stream
    cap = cv2.VideoCapture(rtsp_url)

    # Check if the stream is opened successfully
    if not cap.isOpened():
        print(f"Error: Failed to open RTSP stream '{rtsp_url}'.")
        break

    # Read a frame from the stream
    ret, frame = cap.read()

    # Release the video capture object
    cap.release()

    # Check if a frame is read successfully
    if ret:
        # Generate a unique filename based on the current timestamp
        timestamp = time.strftime("%Y%m%d%H%M%S")
        file_path = os.path.join(output_directory, f"screenshot_{timestamp}.jpg")

        # Save the frame as a JPEG file
        cv2.imwrite(file_path, frame)

        print(f"Screenshot saved: {file_path}")

    # Wait for 3 seconds before capturing the next screenshot
    time.sleep(3)

    # Delete old screenshots to save space
    for filename in os.listdir(output_directory):
        file_path = os.path.join(output_directory, filename)
        file_time = os.path.getmtime(file_path)
        if time.time() - file_time >= 3:
            os.remove(file_path)
            print(f"Old screenshot deleted: {file_path}")
