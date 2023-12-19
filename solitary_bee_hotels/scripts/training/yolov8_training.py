from ultralytics import YOLO
import argparse


def main(model_size, data, epochs, device, save):
    # Load a model
    model = YOLO(f'yolov8{model_size}.pt')  # load a pretrained model 

    # Train the model on Apple M1 Computer
    results = model.train(data=data, epochs=epochs, device=device, save=save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")
    parser.add_argument("--model_size", type=str, default="m", help="Size of model to train")
    parser.add_argument("--data", type=str, default="/Users/edwardamoah/Documents/GitHub/BeeVision/solitary_bee_hotels/datasets/solitary-bee-hotels-4/data.yaml", help="Path to dataset yaml file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--device", type=str, default="mps", help="Device to train on")
    parser.add_argument("--save", type=bool, default=True, help="Save model")
    args = parser.parse_args()
    main(args.model_size, args.data, args.epochs, args.device, args.save)
