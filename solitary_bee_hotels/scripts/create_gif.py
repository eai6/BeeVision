import os
import cv2

def create_video(folder_path, output_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            images.append(image_path)

    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 1, (width, height))

    for image_path in images:
        frame = cv2.imread(image_path)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

# Usage example
folder_path = "/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/outputs"
output_path = "video_output.mp4"
create_video(folder_path, output_path)
