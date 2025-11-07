import os
from ultralytics import YOLO

import configuration as cfg


if __name__ == "__main__":
    model = YOLO("yolo11m.pt") # Load a pretrained YOLO model

    data_path = os.path.join(cfg.YOLO_V11_PATH, "dataset.yaml")  # Path to the dataset YAML file

    results = model.train(
        data=data_path,
        epochs=200,
        device=0,
        batch=8,
        workers=4,
        name="bs8_200"
    )