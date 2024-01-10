import cv2
from ultralytics import YOLO
import pandas as pd
import argparse
import os

def main(video, filename, persist):
    # Load the YOLOv8 model
    model = YOLO('models/nest_detection_model.pt')

    # Open the video file
    cap = cv2.VideoCapture(video)

    # Loop through the video frames
    frame_counter = 0
    nest_detections = []
    nest_state = []
    frames = []


    # create folder to save annotated frames
    file = video.split('/')[-1]
    output_folder = f"/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/outputs/nest_frames/{file.split('.')[0]}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while cap.isOpened() and frame_counter < 100:
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

            #   save results to folder
            cv2.imwrite(f"{output_folder}/annotated_frame_{frame_counter}.jpg", annotated_frame)
            cv2.imwrite(f"{output_folder}/frame_{frame_counter}.jpg", frame)

            # get nest detections
            

            # Break the loop if 'q' is pressed
            #if cv2.waitKey(1) & 0xFF == ord("q"):
            #    break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame({'frames': frames, 'detections_coordinates': nest_detections, 'detections_classes': nest_state})
    
    if persist: # if persist is true, save to filename
        df.to_csv(filename, index=False)


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="/Users/edwardamoah/Downloads/2023-05-29_14_20_01.mp4", required=False, help="path to input video file")
    parser.add_argument("--persist", default=True, required=False, help="persist results to csv")
    parser.add_argument("--filename", default="nest_results.csv", required=False, help="filename to persist results to")
    args = parser.parse_args()
    main(args.video, args.filename, args.persist)