import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd

# Load the YOLOv8 model
#model = YOLO('/Users/edwardamoah/Documents/GitHub/BeeVision/runs/detect/train11/weights/best.pt')

#model = YOLO('/Users/edwardamoah/Documents/GitHub/BeeVision/runs/detect/train13/weights/best.pt')

model = YOLO('/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/runs/detect/train14/weights/best.pt')

# Read video file or capture from camera (replace 'your_video.mp4' with your video file)
video1 = '/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/datasets/videos/2023-05-29_14_00_01.mp4'
cap = cv2.VideoCapture(video1)


# Read the first frame
ret, prev_frame = cap.read()

# Convert the frame to grayscale
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

def update_background(current_frame, prev_bg, alpha):
    bg = alpha * current_frame + (1 - alpha) * prev_bg
    bg = np.uint8(bg)  
    return bg

frame_counter = 0
frames = []
motions_cords = []
frame_motions = []
detections_cords = []
frame_detections = []
while cap.isOpened():

    frame_counter += 1
    # Read the next frame

    ret, current_frame = cap.read()
    
    if not ret:
        break

    frames.append(frame_counter)

    
    # Convert the frame to grayscale
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the absolute difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(current_frame_gray, prev_frame_gray)
    
    # Apply thresholding to the difference frame
    _, thresholded_frame = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours in the thresholded frame
    contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw rectangles around moving objects
    frame_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 200:  # You can adjust this threshold based on your specific case
            (x, y, w, h) = cv2.boundingRect(contour)
            #cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame_contours.append((x, y, w, h))

    # find average of all contours
    if len(frame_contours) > 0:
        x = np.mean([x for (x, y, w, h) in frame_contours])
        y = np.mean([y for (x, y, w, h) in frame_contours])
        w = np.mean([w for (x, y, w, h) in frame_contours])
        h = np.mean([h for (x, y, w, h) in frame_contours])
        motions_cords.append((x, y, w, h))
        frame_motions.append(frame_contours)
        cv2.rectangle(current_frame, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 255, 0), 2)
    else: # no motion detection
        motions_cords.append((0, 0, 0, 0))
        frame_motions.append([])
               

    results = model(current_frame)

    # annotated frame from model
    annotated_frame = results[0].plot()

    # get detection 
    boxes = results[0].boxes.xywh.tolist()

    # get detection cords
    model_detections = [(x,y,w,h) for (x,y,w,h) in boxes]

    #for box in boxes:
    #    x, y, w, h = box
    #    model_detections.append((x, y, w, h))

    # find average of all detections
    if len(model_detections) > 0:
        x = np.mean([x for (x, y, w, h) in model_detections])
        y = np.mean([y for (x, y, w, h) in model_detections])
        w = np.mean([w for (x, y, w, h) in model_detections])
        h = np.mean([h for (x, y, w, h) in model_detections])
        detections_cords.append((x, y, w, h))
        frame_detections.append(model_detections)
        cv2.rectangle(annotated_frame, (int(x), int(y)), (int(x) + int(w), int(y) + int(h)), (0, 255, 255), 2)
    else: # no model detection
        detections_cords.append((0, 0, 0, 0))
        frame_detections.append([])
        


    # Run YOLOv8 inference on the frame
    #results = model(current_frame)
    
    # Visualize the results on the frame
    #annotated_frame = results[0].plot()

    #print(results[0].xyxy)

    # Display the resulting frame
    cv2.imshow('Motion Detection & YOLOv8 Inference', annotated_frame)

    #cv2.imshow("YOLOv8 Inference", annotated_frame)
    
    # Update the previous frame
    #prev_frame_gray = current_frame_gray
    prev_frame_gray = update_background(current_frame_gray, prev_frame_gray, 0.1)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close all windows
cap.release()
cv2.destroyAllWindows()

'''
print(f"motions_cords: {len(motions_cords)}")
print(f"detections_cords: {len(detections_cords)}")
print(f"frame_motions: {len(frame_motions)}")
print(f"frame_detections: {len(frame_detections)}")
print(f"frames: {len(frames)}")
'''

# save results to csv
#df = pd.DataFrame({'frame': frames, 'motions': motions_cords, 'detections': detections_cords})
df = pd.DataFrame({'frame': frames, 'motions': motions_cords, 'detections': detections_cords, "frame_motions": frame_motions, "frame_detections": frame_detections})
df.to_csv('results.csv', index=False)
