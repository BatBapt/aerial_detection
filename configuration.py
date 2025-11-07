import os

BASE_PATH = "D:/Programmation/IA/datas/drone_dataset"

YOLO_PATH = os.path.join(BASE_PATH, "drone-detection-new.v5-new-train.yolov8")
YOLO_V11_PATH = os.path.join(BASE_PATH, "full_dataset_yolo")
COCO_PATH = os.path.join(BASE_PATH, "coco_json_drone_detection")
RESULT_DIR = "results"

BASE_TRAINING_PATH = "runs/detect"
MODEL_WEIGHTS = {
    "bs8_100": os.path.join(BASE_TRAINING_PATH, "bs8_100/weights/best.pt"),
    "bs4_100": os.path.join(BASE_TRAINING_PATH, "bs4_100/weights/best.pt"),
    "bs4_200": os.path.join(BASE_TRAINING_PATH, "bs4_200/weights/best.pt"),
}

LABEL_NOMENCLATURE = {
    0: "AirPlane",
    1: "Drone",
    2: "Helicopter"
}