from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data='/Users/edwardamoah/Documents/GitHub/BeeVision/datasets/Solitary-Bees-3/data.yaml', epochs=10, device='mps', save=True)

# save the model
model.export("models/custom_model_v3.pt")