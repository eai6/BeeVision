# BeeVision - Solitary Bee Hotels

This project aims to create a system for monitoring and analyzing solitary bee hotels. Solitary bee hotels are artificial structures designed to provide nesting sites for solitary bees, which are important pollinators.

## Subfolders

The project is organized into the following subfolders:

- `scripts`: Contains the source code for the BeeVision system. Inlcuding the scripts for training the yolo models
- `datasets`: Stores the video data collected from the bee hotels.
- `outputs`: Temporal storage for program outputs like media etc
- `runs`: Yolo model training output files. This is where the model is saved during training
- `models`: Contain trained Yolo model for nest detection and motion tracking of bees

Feel free to explore each subfolder for more details on their contents and purpose.

## Getting Started

The steps for working with the system inludes 
1. Processes videos to extract images for annotation. The scripts under the data_processsing folder can be used to for this. 
2. Upload Images onto the RoboFlow platform for annotation. https://app.roboflow.com/beevision/solitary-bee-hotels/overview
3. Annotate the images using the RoboFlow platform
4. Export annotated images in the yolov8 format for training. The scripts under the data_processing folder can be used for this. 
5. Train the yolov8 model using the script under the training folder. You will need to specify the --model_size, training --epochs, and --data path to the data.yaml file for the exported annotated data.
6. Execute the motion_tracking.py script to extract bee tracking information. 

## Contributing

If you would like to contribute to the BeeVision project, please follow go ahead and folk the project.

## License

This project is licensed under the [MIT License](LICENSE). Please review the license file for more information.
