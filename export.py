from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('/Users/edwardamoah/Documents/GitHub/corn-earworm/output/models/best.pt')  # load a custom trained model

# Export the model
model.export(format='tflite')