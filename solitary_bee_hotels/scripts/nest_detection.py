import cv2
from ultralytics import YOLO
import pandas as pd

# Load the YOLOv8 model
model = YOLO('models/nest_detection_model.pt')

# Open the video file
video_path = "/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/datasets/videos/2023-09-16/2023-09-16_10_30_00.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
frame_counter = 0
nest_detections = []
nest_state = []
frames = []
while cap.isOpened() :#and frame_counter < 10:
    frame_counter += 1
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # get detection 
        boxes = results[0].boxes.xywh.tolist()
        nest_detections.append(boxes)

        # nest labels
        labels = results[0].boxes.cls.tolist()
        nest_state.append(labels)

        frames.append(frame_counter)

        # get nest detections
        

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()


df = pd.DataFrame({'frames': frames, 'detections': nest_detections, 'state': nest_state})
df.to_csv('nest_results.csv', index=False)