import cv2
from datetime import datetime, timedelta
import ast
import argparse
import pandas as pd 


def main(video, start, motion, nests, output):

    # Load the motion tracking data
    motion = pd.read_csv(motion)

    # Load the nest identification data
    nests = pd.read_csv(nests)

    # Define the start time of the video
    video_start_time = start
    # Load the video
    video = cv2.VideoCapture(video)

    # Define the output video path
    #output_video_path = "/path/to/output/video.mp4"

    #output_video_path = "/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/outputs/video/video2.mp4"

    output_video_path = output

    # Define the video codec and FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the annotated frames as a video file
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    def get_activity_period(frame_number):
        for index, row in motion.iterrows():
            activity_period = row["activity_period_id"]
            frames = ast.literal_eval(row["frames"])
            if frame_number in range(frames[0], frames[1]+1):
                return activity_period
        return -1 # no activity period found

    def frames_to_seconds(frames):
        seconds = frames / 30
        return seconds

    def increment_timestamp(timestamp, seconds):
        # Convert the timestamp string to a datetime object
        dt = datetime.strptime(timestamp, '%H:%M:%S')
        
        # Increment the datetime object by the specified number of seconds
        dt += timedelta(seconds=seconds)
        
        # Convert the datetime object back to a timestamp string
        incremented_timestamp = dt.strftime('%H:%M:%S')
        
        return incremented_timestamp

    # Iterate over each frame in the video
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # Get the frame number
        frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))


        # Plot the nest IDs on the frame
        for index, row in nests.iterrows():
            nest_id = row["nest_id"]
            coordinates = ast.literal_eval(row["coordinates"])
            x, y, w, h = coordinates
            cv2.putText(frame, f"{nest_id}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Check if the frame is within the motion frames
        if get_activity_period(frame_number) != -1:
            # Get the corresponding motion information
            activity_period = get_activity_period(frame_number)
            motion_info = motion[motion["activity_period_id"] == activity_period].iloc[0]
            action = motion_info["action"]
            nest_ids = motion_info["nest_ids"]
            #timestamps = motion_info["timestamp"]
            timestamps = frames_to_seconds(frame_number)
            timestamps = increment_timestamp(video_start_time, timestamps)
            class_ = motion_info["species"]
            
            # Add the motion information to the frame
            cv2.putText(frame, f"Action: {action}", (825, 580), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Timestamp: {timestamps}", (825, 620), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Nest IDs: {nest_ids}", (825, 660), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Species: {class_}", (825, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            
        
        # Write the annotated frame to the output video
        output_video.write(frame)

    # Release the video and output video objects
    video.release()
    output_video.release()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process video, video processing and nest data to create a visualisation of the video')
    parser.add_argument('--video', default="/Users/edwardamoah/Downloads/2023-05-29_14_20_01.mp4", help='path to the original video file')
    parser.add_argument('--start', default="14:20:02", help='start time of video')
    parser.add_argument('--motion', default="/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/final_output.csv",help='path to synthesized video processing csv file')
    parser.add_argument('--nests', default='/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/nest_output.csv',help='path to nests identification csv data file')
    parser.add_argument('--output', default="/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/outputs/video/video3.mp4", help='path to output video file to save the annotated video')
    args = parser.parse_args()

    main(args.video, args.start, args.motion, args.nests, args.output)