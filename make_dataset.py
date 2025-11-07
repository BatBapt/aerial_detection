import os
import shutil
import random

import configuration as cfg

def create_full_dataset(src_path, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    labels_dir = os.path.join(dst_path, 'labels')
    images_dir = os.path.join(dst_path, 'images')

    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    split_name = ["train", "valid", "test"]

    for split in split_name:
        src_split_path = os.path.join(src_path, split)

        if split == "valid":
            split = "val"

        labels_dir_split = os.path.join(labels_dir, split)
        if not os.path.exists(labels_dir_split):
            os.makedirs(labels_dir_split)

        images_dir_split = os.path.join(images_dir, split)
        if not os.path.exists(images_dir_split):
            os.makedirs(images_dir_split)

        print(f"Processing split: {split}")

        images_path = os.path.join(src_split_path, "images")
        labels_path = os.path.join(src_split_path, "labels")

        for i, image in enumerate(os.listdir(images_path)):
            if image.endswith('.jpg'):
                label_file = image.replace('.jpg', '.txt')
                src_image_path = os.path.join(images_path, image)
                src_label_path = os.path.join(labels_path, label_file)

                dst_image_path = os.path.join(images_dir_split, image)
                dst_label_path = os.path.join(labels_dir_split, label_file)

                if os.path.exists(dst_image_path) and os.path.exists(dst_label_path):
                    continue

                shutil.copy(src_image_path, dst_image_path)
                shutil.copy(src_label_path, dst_label_path)

                if i % 100 == 0:
                    print(f"\tCopied {i} images and labels to {split} set")

    dataset_yaml_path = os.path.join(full_dataset_path, "dataset.yaml")
    with open(dataset_yaml_path, 'w') as f:
        f.write("path: D:/Programmation/IA/datas/drone_dataset/full_dataset_yolo/\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("nc: 3\n")  # Number of classes
        f.write("names: ['AirPlane', 'Drone', 'Helicopter']\n")  # Class names
    print(f"Dataset YAML file created at {dataset_yaml_path}")

if __name__ == "__main__":
    original_path = cfg.YOLO_PATH
    full_dataset_path = cfg.YOLO_V11_PATH
    create_full_dataset(original_path, full_dataset_path)

