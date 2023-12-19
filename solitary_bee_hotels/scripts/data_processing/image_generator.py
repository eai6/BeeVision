import argparse
import cv2
import numpy as np
import os

def main(video, output_dir, sensitivity):
    # Read video file or capture from camera (replace 'your_video.mp4' with your video file)

    cap = cv2.VideoCapture(video)

    # get filename
    filename = video.split('/')[-1].split('.')[0]

    # make output directory for video if it doesn't exist
    folder_name = f"{output_dir}/{filename}_{sensitivity}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Read the first frame
    ret, prev_frame = cap.read()

    # Convert the frame to grayscale
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Counter for saved frames
    frame_counter = 0

    def update_background(current_frame, prev_bg, alpha):
        bg = alpha * current_frame + (1 - alpha) * prev_bg
        bg = np.uint8(bg)  
        return bg

    while cap.isOpened():

        frame_counter += 1

        # Read the next frame
        ret, current_frame = cap.read()
        
        if not ret:
            break
        
        # Convert the frame to grayscale
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate the absolute difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(current_frame_gray, prev_frame_gray)
        
        # Apply thresholding to the difference frame
        _, thresholded_frame = cv2.threshold(frame_diff, sensitivity, 255, cv2.THRESH_BINARY)
        
        # Find contours in the thresholded frame
        contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Save frames with motion as images
        for contour in contours:
            if cv2.contourArea(contour) > 200:  # You can adjust this threshold based on your specific case
                # Save the frame with motion as an image
                frame_name = f"{folder_name}/{filename}_frame_{frame_counter}.jpg"
                cv2.imwrite(frame_name, current_frame)
                print(f"Saved {frame_name}")
                
                break
        
        # adaptive background update
        prev_frame_gray = update_background(current_frame_gray, prev_frame_gray, 0.1)
        


    # Release the capture object 
    cap.release()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frames from videos when motion is detected')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output_dir', type=str, default='/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/datasets/images', help='Path to output directory')
    parser.add_argument('--sensitivity', type=int, default=50, help='Sensitivity of motion detection')
    main(**vars(parser.parse_args()))