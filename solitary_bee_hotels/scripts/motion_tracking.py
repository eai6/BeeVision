import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
import argparse
import os


def main(video, persist, filename):
        
    # Load the YOLOv8 model
    model = YOLO('models/bee_detection_model.pt')

    # Replace with video path that you want to do motion tracking on
    #video5 = "/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/datasets/videos/annotated/2023-09-08_10_10_01.mp4"
    cap = cv2.VideoCapture(video)


    # Read the first frame
    ret, prev_frame = cap.read()

    # Convert the frame to grayscale
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    def update_background(current_frame, prev_bg, alpha):
        '''
        Update background using exponential weighted average
        Exponential weighted average formula:
        current_frame = alpha * current_frame + (1 - alpha) * prev_bg
        
        '''
        bg = alpha * current_frame + (1 - alpha) * prev_bg
        bg = np.uint8(bg)  
        return bg

    frame_counter = 0
    frames = []
    motions_cords = []
    frame_motions = []
    detections_cords = []
    frame_detections = []
    class_ids = []


    # create folder to save annotated frames
    file = video.split('/')[-1]
    output_folder = f"/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/outputs/tracking_frames/{file.split('.')[0]}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while cap.isOpened():

        frame_counter += 1
        # Read the next frame

        ret, current_frame = cap.read()
        
        if not ret:
            break

        frames.append(frame_counter)
        
        # Convert the frame to grayscale
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)


        if frame_counter == 1:
            background_frame_gray = current_frame_gray
        
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
                cv2.rectangle(current_frame, (x, y), (x + w, y + h), (255,0, 0), 2)
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
                

        # Run YOLOv8 inference on the frame if motion is detected
        if len(frame_contours) > 0:
            results = model(current_frame)

            # annotated frame from model
            annotated_frame = results[0].plot()

            # get detection 
            boxes = results[0].boxes.xywh.tolist()

            #track_ids = results[0].boxes.id.int().cpu().tolist()

            # get detection cords
            model_detections = [(x,y,w,h) for (x,y,w,h) in boxes]

            #for box in boxes:
            #    x, y, w, h = box
            #    model_detections.append((x, y, w, h))

            # get class ids
            labels = results[0].boxes.cls.tolist()
            class_ids.append(labels)

            # find average of all detections
            if len(model_detections) > 0:
                x = np.mean([x for (x, y, w, h) in model_detections])
                y = np.mean([y for (x, y, w, h) in model_detections])
                w = np.mean([w for (x, y, w, h) in model_detections])
                h = np.mean([h for (x, y, w, h) in model_detections])
                detections_cords.append((x, y, w, h))
                frame_detections.append(model_detections)
                cv2.rectangle(annotated_frame, (int(x) - int(w/2), int(y) - int(h/2)), (int(x) + int(w/2), int(y) + int(h/2)), (0, 255, 255), 2)
            else: # no model detection
                detections_cords.append((0, 0, 0, 0))
                frame_detections.append([])
        else: # no motion detection
            detections_cords.append((0, 0, 0, 0))
            frame_detections.append([])

            # add class ids
            class_ids.append([])
            

        # Display the resulting frame
        if len(frame_contours) > 0:
            cv2.imshow('Motion Detection & YOLOv8 Inference', annotated_frame)
            # save annotated frame
            cv2.imwrite(f"{output_folder}/video_{frame_counter}.jpg", annotated_frame)
        else:
            cv2.imshow('Motion Detection & YOLOv8 Inference', current_frame)

        #cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        # Update the previous frame
        #prev_frame_gray = current_frame_gray
        #prev_frame_gray = update_background(current_frame_gray, prev_frame_gray, 0.1)
        prev_frame_gray = background_frame_gray
        
        # Break the loop if 'q' key is pressed
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # Release the capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


    # save results to csv
    #df = pd.DataFrame({'frame': frames, 'motions': motions_cords, 'detections': detections_cords})
    df = pd.DataFrame({'frame_number': frames, 'motions_coordinates': motions_cords, 'detections_coordinates': detections_cords, 'detections_classes': class_ids, "frame_motions": frame_motions, "frame_detections": frame_detections})
    
    if persist: # if persist is true, save to filename
        df.to_csv(filename, index=False)


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/datasets/videos/annotated/2023-09-08_10_10_01.mp4", required=False, help="path to input video file")
    parser.add_argument("--persist", default=True, required=False, help="persist results to csv")
    parser.add_argument("--filename", default="results.csv", required=False, help="filename to persist results to")
    args = parser.parse_args()
    main(args.video, args.persist, args.filename)
