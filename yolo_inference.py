from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO(r"C:\Users\bethi\OneDrive\Desktop\coco\best_yolo.pt")

# Open video file
video_path = r"C:\Users\bethi\OneDrive\Desktop\coco\traffic.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Video file not found or cannot be opened!")
    exit()

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit when video ends

    # Run YOLO on the current frame
    results = model.predict(frame, conf=0.5)

    # Extract the frame with detected objects
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLO Object Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and close OpenCV windows
cap.release()
cv2.destroyAllWindows()