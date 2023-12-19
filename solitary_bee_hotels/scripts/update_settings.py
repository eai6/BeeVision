from ultralytics import settings
import argparse

def main(run_dir, datasets_dir, weights_dir):

    # Update multiple settings
    settings.update({
        'runs_dir': run_dir, 
        'datasets_dir': datasets_dir,
        'weights_dir': weights_dir
    })

    # Reset settings to default values
    print(settings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Update yolo settings")
    parser.add_argument("--run_dir", type=str, default="/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/runs", help="Path to run directory")
    parser.add_argument("--datasets_dir", type=str, default="/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/datasets", help="Path to datasets directory")
    parser.add_argument("--weights_dir", type=str, default="/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/weights", help="Path to weights directory")
    args = parser.parse_args()
    main(args.run_dir, args.datasets_dir, args.weights_dir)