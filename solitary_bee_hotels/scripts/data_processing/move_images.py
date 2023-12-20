import os
import shutil
import argparse


def main(images_paths, dest_path):
    #images_paths = "/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/datasets/select_images"
    folders = os.listdir(images_paths)
    folders = [os.path.join(images_paths, folder) for folder in folders]
    folders = [folder for folder in folders if os.path.isdir(folder)]
    #folders
    #dest_path = "/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/datasets/roboflow_images"
    for folder in folders:
        print(f"Moving images in {folder}")
        files = os.listdir(folder)
        #files = [os.path.join(folder, file) for file in files]
        for file in files:
            dest = os.path.join(dest_path, file)
            org = os.path.join(folder, file)
            shutil.copy(org, dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move selected images to a new folder")
    parser.add_argument(
        "--images_paths",
        type=str,
        help="Path to the folder containing the images folders",
        default="/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/datasets/select_images",
    )
    parser.add_argument(
        "--dest_path",
        type=str,
        help="Path to the folder where the images will be moved to",
        default="/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/datasets/roboflow_images",
    )
    args = parser.parse_args()
    main(args.images_paths, args.dest_path)