import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
#model = YOLO('/Users/edwardamoah/Documents/GitHub/BeeVision/runs/detect/train11/weights/best.pt')

#model = YOLO('/Users/edwardamoah/Documents/GitHub/BeeVision/runs/detect/train13/weights/best.pt')

model = YOLO('/Users/edwardamoah/Documents/GitHub/BeeVision/runs/detect/train14/weights/best.pt')

# Read video file or capture from camera (replace 'your_video.mp4' with your video file)
video1 = '/Users/edwardamoah/Documents/GitHub/BeeVision/datasets/videos/2023-05-10_17_46_37.mp4'
video2 = "/Users/edwardamoah/Documents/GitHub/BeeVision/datasets/videos/2023-05-29_16_00_01.mp4"
video3 = "/Users/edwardamoah/Documents/GitHub/BeeVision/datasets/videos/2023-05-10_18_40_01.mp4"
video4 = '/Users/edwardamoah/Documents/GitHub/BeeVision/datasets/videos/2023-05-10_17_50_01.mp4'
cap = cv2.VideoCapture(video1)


# Read the first frame
ret, prev_frame = cap.read()

# Convert the frame to grayscale
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    # Read the next frame
    ret, current_frame = cap.read()
    
    if not ret:
        break
    
    # Convert the frame to grayscale
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the absolute difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(current_frame_gray, prev_frame_gray)
    
    # Apply thresholding to the difference frame
    _, thresholded_frame = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded frame
    contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw rectangles around moving objects
    for contour in contours:
        if cv2.contourArea(contour) > 200:  # You can adjust this threshold based on your specific case
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Run YOLOv8 inference on the frame
    results = model(current_frame)
    
    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    #print(results[0].xyxy)

    # Display the resulting frame
    #cv2.imshow('Motion Detection', current_frame)

    cv2.imshow("YOLOv8 Inference", annotated_frame)
    
    # Update the previous frame
    prev_frame_gray = current_frame_gray
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close all windows
cap.release()
cv2.destroyAllWindows()
