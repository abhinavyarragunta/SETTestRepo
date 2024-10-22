import cv2
from ultralytics import YOLO

# Load the pre-trained YOLOv11 model for pose estimation
model = YOLO('yolo11n-pose.pt')  # Replace with the actual pose estimation model

# Open a connection to the webcam (0 is typically the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # If frame capture was unsuccessful, break the loop
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run pose estimation on the captured frame
    results = model(frame)

    # Visualize the results on the frame
    result_frame = results[0].plot()

    # Display the frame with the pose estimation overlay
    cv2.imshow('YOLOv11 Pose Estimation', result_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()
