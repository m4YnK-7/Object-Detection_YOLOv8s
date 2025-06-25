from ultralytics import YOLO


model = YOLO("best_model.pt") 

# Export to CoreML format
model.export(
    format="coreml",
    half=True,
    nms=True,

    imgsz=128,
    data="utils\yolo_params.yaml",
    device="cpu"
    )