import os
import cv2
import re
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import ast

import configuration as cfg


def load_annot_file(file_path):
    content = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                content.append(line)
    return content


def process_true_labels(image, labels):
    label, x_center, y_center, width, height = map(float, labels.split())
    label = int(label)

    x_min = int((x_center - width / 2) * image.shape[1])
    y_min = int((y_center - height / 2) * image.shape[0])
    x_max = int((x_center + width / 2) * image.shape[1])
    y_max = int((y_center + height / 2) * image.shape[0])

    return x_min, y_min, x_max, y_max, label


def process_pred_boxes(boxes):
    processed_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        processed_boxes.append({
            "predicted_boxes": [x1, y1, x2, y2],
            "confidence": conf,
            "class": cls
        })
    return processed_boxes


def draw_boxes_true(image, boxes, thickness=2):
    """Dessine les boîtes englobantes sur l'image."""
    for box in boxes:

        x_min, y_min, x_max, y_max, label = process_true_labels(image, box)

        if label == 0:
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness)
            cv2.putText(image, cfg.LABEL_NOMENCLATURE[label], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), thickness)
            cv2.putText(image, cfg.LABEL_NOMENCLATURE[label], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    return image


def draw_boxes_predict(image, boxes, names, thickness=2):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = box.cls[0]
        label = f"{names[int(cls)]}: {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness)
    return image


def compute_iou(box_true, box_pred):
    """Calcule l'IoU entre deux boîtes (x1, y1, x2, y2)"""
    try:
        x1_true, y1_true, x2_true, y2_true = box_true
        x1_pred, y1_pred, x2_pred, y2_pred = box_pred
    except TypeError:
        return 0.0

    # Calcul des coordonnées de l'intersection
    x1_inter = max(x1_true, x1_pred)
    y1_inter = max(y1_true, y1_pred)
    x2_inter = min(x2_true, x2_pred)
    y2_inter = min(y2_true, y2_pred)

    # Aire de l'intersection
    inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    # Aire des deux boîtes
    true_area = (x2_true - x1_true + 1) * (y2_true - y1_true + 1)
    pred_area = (x2_pred - x1_pred + 1) * (y2_pred - y1_pred + 1)

    # Aire de l'union
    union_area = true_area + pred_area - inter_area

    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def compute_dice(box_true, box_pred):
    x1_1, y1_1, x2_1, y2_1 = box_true
    x1_2, y1_2, x2_2, y2_2 = box_pred

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        intersection_area = 0.0
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

    union_area = area1 + area2 - intersection_area

    if union_area == 0:
        return 0.0

    dice = (2.0 * intersection_area) / (area1 + area2)

    return dice


def load_and_convert_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"File {csv_path} is empty.")
        return pd.DataFrame()

    # Convertir les colonnes bbox de chaînes en tuples/listes
    df['true_bbox'] = df['true_bbox'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['predicted_bbox'] = df['predicted_bbox'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    return df


def calculate_tp_fp_fn_tn(df, iou_threshold=0.95, conf_threshold=0.5):
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    true_negatives = defaultdict(int)
    iou_scores = defaultdict(list)

    # Filtrer les prédictions par seuil de confiance
    df_filtered = df[df['confidence'] >= conf_threshold]

    # Pour les images AVEC annotation
    matched_true_indices = set()
    matched_pred_indices = set()

    for i, true_row in df[df['true_class'].notna()].iterrows():
        for j, pred_row in df_filtered.iterrows():
            if j in matched_pred_indices:
                continue
            if i == j:
                iou = compute_iou(true_row['true_bbox'], pred_row['predicted_bbox'])
                if iou >= iou_threshold and true_row['true_class'] == pred_row['predicted_class']:
                    true_positives[true_row['true_class']] += 1
                    matched_true_indices.add(i)
                    matched_pred_indices.add(j)
                    iou_scores[true_row['true_class']].append(iou)
                    break

    # False Negatives = boîtes vraies non associées
    for i, true_row in df[df['true_class'].notna()].iterrows():
        if i not in matched_true_indices:
            false_negatives[true_row['true_class']] += 1

    # False Positives = boîtes prédites non associées (sur images avec ou sans annotation)
    for j, pred_row in df_filtered.iterrows():
        if j not in matched_pred_indices:
            false_positives[pred_row['predicted_class']] += 1

    # True Negatives = images SANS annotation ET sans prédiction
    for i, row in df[df['true_class'].isna()].iterrows():
        if pd.isna(row['predicted_class']):
            true_negatives['background'] += 1

    return true_positives, false_positives, false_negatives, true_negatives, iou_scores



def calculate_metrics(tp, fp, fn, tn, class_id):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    return precision, recall, f1, accuracy

def calculate_all_metrics(df, iou_threshold=0.95, conf_threshold=0.5):
    tp, fp, fn, tn, iou_scores = calculate_tp_fp_fn_tn(df, iou_threshold, conf_threshold)
    metrics = {}

    # Métriques par classe
    all_classes = set(tp.keys()).union(set(fp.keys())).union(set(fn.keys()))
    for class_id in all_classes:
        tp_class = tp.get(class_id, 0)
        fp_class = fp.get(class_id, 0)
        fn_class = fn.get(class_id, 0)
        tn_class = tn.get('background', 0)  # TN est global pour toutes les classes (images vides)
        precision, recall, f1, accuracy = calculate_metrics(tp_class, fp_class, fn_class, tn_class, class_id)
        metrics[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'iou_mean': np.mean(iou_scores.get(class_id, [])) if iou_scores.get(class_id, []) else 0,
            'tp': tp_class,
            'fp': fp_class,
            'fn': fn_class,
            'tn': tn_class,
        }

    # Métriques globales
    tp_total = sum(tp.values())
    fp_total = sum(fp.values())
    fn_total = sum(fn.values())
    tn_total = tn.get('background', 0)
    precision_total, recall_total, f1_total, accuracy_total = calculate_metrics(tp_total, fp_total, fn_total, tn_total, 'total')
    metrics['total'] = {
        'precision': precision_total,
        'recall': recall_total,
        'f1': f1_total,
        'accuracy': accuracy_total,
        'iou_mean': np.mean([iou for scores in iou_scores.values() for iou in scores]) if iou_scores else 0,
        'tp': tp_total,
        'fp': fp_total,
        'fn': fn_total,
        'tn': tn_total,
    }

    return metrics


def display_metrics(metrics):
    for class_id, metric in metrics.items():
        if class_id != 'total':
            print(f"\tClass {class_id}:")
            print(f"\t\tPrecision: {metric['precision']:.4f}, Recall: {metric['recall']:.4f}, F1: {metric['f1']:.4f}, Accuracy: {metric['accuracy']:.4f}")
            print(f"\t\tTP: {metric['tp']}, FP: {metric['fp']}, FN: {metric['fn']}, TN: {metric['tn']}")
    total = metrics['total']
    print("\tTotal:")
    print(f"\t\tPrecision: {total['precision']:.4f}, Recall: {total['recall']:.4f}, F1: {total['f1']:.4f}, Accuracy: {total['accuracy']:.4f}")
    print(f"\t\tTP: {total['tp']}, FP: {total['fp']}, FN: {total['fn']}, TN: {total['tn']}")


def load_data(file_paths):
    data_frames = []
    for path in file_paths:
        df = pd.read_csv(path)
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

def extract_model_name_and_conf(dir_name):
    match = re.search(r'(.+)_conf_(\d+\.?\d*)', dir_name)
    if match:
        model_name = match.group(1)
        conf_threshold = float(match.group(2))/100
        return model_name, conf_threshold
    return None, None

def load_and_evaluate(file_path, conf_threshold, iou_threshold):
    """
    Charge et évalue les résultats pour un fichier donné, en calculant toutes les métriques (precision, recall, F1, accuracy, IoU, Dice).
    """
    try:
        data = pd.read_csv(file_path)
        # Filtrer les prédictions par seuil de confiance et IoU
        filtered_data = data[(data['confidence'] >= conf_threshold) & (data['iou'] >= iou_threshold)]

        # Calculer TP, FP, FN, TN
        tp = len(filtered_data[filtered_data['is_FP'] == False])
        fp = len(data[data['is_FP'] == True])
        fn = len(data[data['is_FN'] == True])
        tn = len(data[data['is_TN'] == True])

        # Calcul des métriques
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        avg_iou = filtered_data['iou'].mean() if len(filtered_data) > 0 else 0
        avg_dice = filtered_data['dice'].mean() if len(filtered_data) > 0 else 0

        # Nombre total d'exemples
        total_examples = len(data)

        return {
            'model_name': os.path.basename(os.path.dirname(file_path)).split('_conf_')[0],
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'avg_iou': avg_iou,
            'avg_dice': avg_dice,
            'num_detections': len(filtered_data),
            'total_examples': total_examples,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def evaluate_all_files(root_dir, iou_thresholds, output_file):
    """
    Évalue tous les fichiers CSV dans les sous-répertoires de root_dir.
    """
    results = []
    total_examples = None

    # Déterminer le nombre total d'exemples à partir du premier fichier CSV valide
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(dir_path, file_name)
                data = pd.read_csv(file_path)
                total_examples = len(data)
                break
        if total_examples is not None:
            break

    if total_examples is None:
        raise ValueError("Aucun fichier CSV trouvé pour déterminer le nombre total d'exemples.")
    # Parcourir chaque sous-répertoire
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        model_name, conf_threshold_dir = extract_model_name_and_conf(dir_name)
        if model_name is None:
            continue

        # Parcourir chaque fichier dans le répertoire
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(dir_path, file_name)
                # Extraire le seuil d'IoU du nom de fichier
                iou_match = re.search(r'iou_(\d+\.?\d*)', file_name)
                if iou_match:
                    iou_threshold = float(iou_match.group(1))/100
                    if iou_threshold in iou_thresholds:
                        result = load_and_evaluate(file_path, conf_threshold_dir, iou_threshold)
                        if result:
                            results.append(result)

    # Sauvegarder les résultats dans un DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Les résultats ont été enregistrés dans {output_file}")
    return results_df

def plot_metrics_from_dataframe(data, output_plot_dir):
    """
    Trace les courbes des métriques (precision, recall, F1, accuracy, IoU, Dice) en fonction du seuil de confiance.
    """
    if not os.path.exists(output_plot_dir):
        os.makedirs(output_plot_dir)

    metrics = ['precision', 'recall', 'f1_score', 'accuracy', 'avg_iou', 'avg_dice']

    for metric in metrics:
        plt.figure(figsize=(12, 6))
        for model_name in data['model_name'].unique():
            subset = data[data['model_name'] == model_name]
            for iou_threshold in subset['iou_threshold'].unique():
                sub_subset = subset[subset['iou_threshold'] == iou_threshold]
                plt.plot(
                    sub_subset['conf_threshold'],
                    sub_subset[metric],
                    marker='o',
                    label=f'{model_name}, IoU={iou_threshold}'
                )
        plt.xlabel('Confidence Threshold')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} vs Confidence Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_plot_dir, f'{metric}_vs_confidence.png'))
        plt.close()


if __name__ == "__main__":

    labels_path = os.path.join(cfg.YOLO_V11_PATH, "labels", "test")

    unique_label = set()

    for file in os.listdir(labels_path):
        file_path = os.path.join(labels_path, file)
        annotations = load_annot_file(file_path)

        if len(annotations) == 0:
            print(file_path)

        for annot in annotations:
            label_id = int(annot.split()[0])
            unique_label.add(label_id)

    print("Unique labels in validation set:", unique_label)

