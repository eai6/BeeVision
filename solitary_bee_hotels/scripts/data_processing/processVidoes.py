import argparse
import os

# default values
output_dir = "/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/datasets/images"
videos = "/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/datasets/videos"

def main(videos, output_dir, sensitivity):
    videos = [f"{videos}/{video}" for video in os.listdir(videos)]  # get all videos in the videos folder
    for video in videos:
        try:
            os.system(f"python3 scripts/data_processing/image_generator.py --video {video} --output_dir {output_dir} --sensitivity {sensitivity}")
        except:
            print(f"Error processing {video}")
            continue
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos to generate images for training. Extracts frames from videos when motion is detected.")
    parser.add_argument("--videos", type=str, default=videos, help="Path to video folder")
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Path to output directory")
    parser.add_argument("--sensitivity", type=int, default=50, help="Sensitivity of motion detection")
    args = parser.parse_args()
    main(args.videos, args.output_dir, args.sensitivity)
