import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('/Users/edwardamoah/Documents/GitHub/BeeVision/runs/detect/train11/weights/best.pt')


# Define the path to the image file
image_path = '/Users/edwardamoah/Documents/GitHub/BeeVision/datasets/images/motion_frame_0.jpg'

# Load the image
image = cv2.imread(image_path)

# Run YOLOv8 inference on the image
results = model(image)

# Visualize the results on the frame
annotated_frame = results[0].plot()

# Display the image

#cv2.imshow('Image', image)
cv2.imshow('YOLOv8 Inference', annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

