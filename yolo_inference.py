from ultralytics import YOLO
import cv2
model=YOLO('best.pt')

# #Image file
# source="img1.png"
# confidence_threshold=0.5
# results=model.predict(source=source,conf=confidence_threshold,show=True)
# annotated_image = results[0].plot()
# # Define output path
# output_path = "output_img1.png"
# # Save the processed image
# cv2.imwrite(output_path, annotated_image)



# Open video file
video_path = "video1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("outputVideo1.mp4", fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when video ends
    
    # Run YOLO on the frame
    results = model.predict(frame, conf=0.5)

    # Ensure correct frame size
    annotated_frame = results[0].plot()
    annotated_frame = cv2.resize(annotated_frame, (frame_width, frame_height))  # 🔹 Fix: Resize to original size

    # Show the processed frame
    cv2.imshow("YOLO Detection", annotated_frame)
    
    # Write frame to output video
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to exit

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()