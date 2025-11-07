import os
from ultralytics import YOLO

import configuration as cfg


if __name__ == "__main__":

    yolo_model_names = {
        "nano": "yolo11n.pt",
        "small": "yolo11s.pt",
        "medium": "yolo11m.pt",
        "large": "yolo11l.pt",
        "extra": "yolo11x.pt"
    }
    type_model = "medium"  # Choose model type: nano, small, medium, large, extra
    model = YOLO(yolo_model_names[type_model])  # Load a pretrained YOLO model

    data_path = os.path.join(cfg.YOLO_V11_PATH, "dataset.yaml")  # Path to the dataset YAML file

    epochs = 200
    device = 0 # Use GPU 0
    batch_size = 8
    workers = 4  # Number of data loading workers
    name = "bs8_200"  # Name for the training run

    results = model.train(
        data=data_path,
        epochs=epochs,
        device=device,
        batch=batch_size,
        workers=workers,
        name=name
    )