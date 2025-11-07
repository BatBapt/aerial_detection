import os
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import pandas as pd

import tools as tools
import configuration as cfg


def predict_n(model, images_path, labels_path, threshold, n_sample=5, outputs_path="results/"):
    images = os.listdir(images_path)
    rd_ix = np.random.randint(0, len(images) - 1, n_sample)

    if not os.path.exists(outputs_path):
        os.makedirs(outputs_path)

    for idx in rd_ix:
        image_name = images[idx]
        label_sample = os.path.join(labels_path, image_name.replace(".jpg", ".txt"))

        image_path = os.path.join(images_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tmp = image.copy()
        labels = tools.load_annot_file(label_sample)
        image_true = tools.draw_boxes_true(image, labels, thickness=1)

        results = model([image_path], conf=threshold)  # return a list of Results objects

        # Get the predicted image (as a numpy array)
        result = results[0]
        image_predict = tools.draw_boxes_predict(image_tmp, result.boxes, names=result.names, thickness=1)

        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(15, 8))
        axs[0].imshow(image_true)
        axs[0].set_title("True")
        axs[0].axis('off')

        axs[1].imshow(image_predict)
        axs[1].set_title("Predict")
        axs[1].axis('off')

        plt.tight_layout()
        plt.savefig(f"{outputs_path}/comparison_{image_name}")
        plt.close(fig)  # Close the figure to free memory

def predict_all(model, images_path, labels_path, output_csv_path, threshold):
    images = os.listdir(images_path)
    nb_images = len(images)
    all_results = []

    for i, image_name in enumerate(images):
        label_sample = os.path.join(labels_path, image_name.replace(".jpg", ".txt"))
        image_path = os.path.join(images_path, image_name)
        labels = tools.load_annot_file(label_sample)

        # Charger l'image pour la prédiction
        results = model([image_path], conf=threshold, verbose=False)
        result = results[0]
        pred_boxes_processed = tools.process_pred_boxes(result.boxes)

        # Cas 1: Image SANS annotation (fichier vide)
        if len(labels) == 0:
            if len(pred_boxes_processed) == 0:
                # Vrai négatif (TN)
                all_results.append({
                    "image_name": image_name,
                    "true_class": None,
                    "true_bbox": None,
                    "predicted_class": None,
                    "predicted_bbox": None,
                    "confidence": 0.0,
                    "is_TN": True,
                    "is_FP": False,
                    "is_FN": False,
                    "iou": None,
                    "dice": None
                })
            else:
                # Faux positif (FP)
                all_results.append({
                    "image_name": image_name,
                    "true_class": None,
                    "true_bbox": None,
                    "predicted_class": pred_boxes_processed[0]["class"],
                    "predicted_bbox": pred_boxes_processed[0]["predicted_boxes"],
                    "confidence": pred_boxes_processed[0]["confidence"],
                    "is_TN": False,
                    "is_FP": True,
                    "is_FN": False,
                    "iou": None,
                    "dice": None
                })
        # Cas 2: Image AVEC annotation
        else:
            true_label = tools.process_true_labels(cv2.imread(image_path), labels[0])
            true_class = true_label[4]
            true_bbox = true_label[:4]

            if len(pred_boxes_processed) == 0:
                # Faux négatif (FN)
                all_results.append({
                    "image_name": image_name,
                    "true_class": true_class,
                    "true_bbox": true_bbox,
                    "predicted_class": None,
                    "predicted_bbox": None,
                    "confidence": 0.0,
                    "is_TN": False,
                    "is_FP": False,
                    "is_FN": True,
                    "iou": 0.0,
                    "dice": 0.0
                })
            else:
                # Calcul des métriques si prédiction présente
                pred_class = pred_boxes_processed[0]["class"]
                pred_bbox = pred_boxes_processed[0]["predicted_boxes"]
                iou = tools.compute_iou(true_bbox, pred_bbox)
                dice = tools.compute_dice(true_bbox, pred_bbox)
                all_results.append({
                    "image_name": image_name,
                    "true_class": true_class,
                    "true_bbox": true_bbox,
                    "predicted_class": pred_class,
                    "predicted_bbox": pred_bbox,
                    "confidence": pred_boxes_processed[0]["confidence"],
                    "is_TN": False,
                    "is_FP": False,
                    "is_FN": False,
                    "iou": iou,
                    "dice": dice
                })

        if (i+1) % 100 == 0:
            print(f"Processed {i+1} / {nb_images} images.")

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_csv_path, index=False)
    return df_results


if __name__ == "__main__":
    # model = YOLO(cfg.BEST_MODEL_WEIGHT)  # Load a pretrained YOLO model
    model_basename = "bs4_100"
    model = YOLO(cfg.MODEL_WEIGHTS[model_basename])

    base_path = cfg.YOLO_V11_PATH
    images_test_path = os.path.join(base_path, "images", "test")
    labels_test_path = os.path.join(base_path, "labels", "test")

    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    conf_thresholds = [0.25, 0.5, 0.75, 0.95]
    ious_threshold = [0.5, 0.75, 0.95]

    for conf_threshold in conf_thresholds:
        print(f"Evaluating at confidence threshold: {conf_threshold}")
        conf_name = int(conf_threshold*100)
        conf_path = os.path.join(output_dir, f"{model_basename}_conf_{conf_name}")
        if not os.path.exists(conf_path):
            os.makedirs(conf_path)

        plot_conf_path = os.path.join(conf_path, "plots")
        if not os.path.exists(plot_conf_path):
            os.makedirs(plot_conf_path)

        nb_sample = 5
        # predict_n(model, images_test_path, labels_test_path, conf_threshold, n_sample=nb_sample, outputs_path=plot_conf_path)

        for iou_threshold in ious_threshold:
            iou_name = int(iou_threshold*100)

            csv_name = f"{model_basename}_prediction_results_conf_{conf_name}_iou_{iou_name}.csv"
            csv_path = os.path.join(conf_path, csv_name)

            if os.path.exists(csv_path):
                results = tools.load_and_convert_csv(csv_path)
            else:
                results = predict_all(model, images_test_path, labels_test_path, csv_path, conf_threshold)

            if results.empty:
                print(f"No predictions were made at confidence {conf_threshold} and IoU {iou_threshold}.")
                continue

            metrics = tools.calculate_all_metrics(results)

            print("*"*75)
            print(f"\tMetrics at IoU threshold: {iou_threshold}")
            tools.display_metrics(metrics)
            print("*"*75)

    print("\nCompute metrics accross all CSV files\n")