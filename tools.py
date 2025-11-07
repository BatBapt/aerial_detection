import os
import cv2
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
import ast

import configuration as cfg


def load_annot_file(file_path):
    # Read an annotation file line-by-line and return non-empty stripped lines
    content = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                content.append(line)
    return content


def process_true_labels(image, labels):
    # Parse a YOLO-format annotation line: label x_center y_center width height (normalized)
    label, x_center, y_center, width, height = map(float, labels.split())
    label = int(label)

    # Convert normalized center/size to pixel bounding box coordinates
    x_min = int((x_center - width / 2) * image.shape[1])
    y_min = int((y_center - height / 2) * image.shape[0])
    x_max = int((x_center + width / 2) * image.shape[1])
    y_max = int((y_center + height / 2) * image.shape[0])

    return x_min, y_min, x_max, y_max, label


def process_pred_boxes(boxes):
    # Convert model prediction objects (with attributes xyxy, conf, cls) into plain dicts
    processed_boxes = []
    for box in boxes:
        # box.xyxy[0] expected to be iterable of (x1, y1, x2, y2)
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
    # Draw ground-truth boxes on image, color-coded per class (0 uses red in this code)
    for box in boxes:
        x_min, y_min, x_max, y_max, label = process_true_labels(image, box)

        if label == 0:
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness)
            cv2.putText(image, cfg.LABEL_NOMENCLATURE[label], (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), thickness)
            cv2.putText(image, cfg.LABEL_NOMENCLATURE[label], (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    return image


def draw_boxes_predict(image, boxes, names, thickness=2):
    # Draw predicted boxes with class name and confidence label
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = box.cls[0]
        label = f"{names[int(cls)]}: {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness)
    return image


def compute_iou(box_true, box_pred):
    # Compute Intersection over Union (IoU) between two boxes given as (x1, y1, x2, y2)
    try:
        x1_true, y1_true, x2_true, y2_true = box_true
        x1_pred, y1_pred, x2_pred, y2_pred = box_pred
    except TypeError:
        # If input is None or incorrectly formatted, return IoU 0
        return 0.0

    # Coordinates of the intersection rectangle
    x1_inter = max(x1_true, x1_pred)
    y1_inter = max(y1_true, y1_pred)
    x2_inter = min(x2_true, x2_pred)
    y2_inter = min(y2_true, y2_pred)

    # Intersection area (add +1 if using inclusive pixel coordinates; here it's included)
    inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    # Areas of each box (inclusive coordinates)
    true_area = (x2_true - x1_true + 1) * (y2_true - y1_true + 1)
    pred_area = (x2_pred - x1_pred + 1) * (y2_pred - y1_pred + 1)

    # Union area and final IoU (guard division by zero)
    union_area = true_area + pred_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def compute_dice(box_true, box_pred):
    # Compute Dice coefficient (F1-score for areas) between two boxes
    x1_1, y1_1, x2_1, y2_1 = box_true
    x1_2, y1_2, x2_2, y2_2 = box_pred

    # Areas of each box (note: here uses non-inclusive width/height)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Overlap rectangle coordinates
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # If no overlap, intersection is zero
    if x_right < x_left or y_bottom < y_top:
        intersection_area = 0.0
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Dice = 2 * intersection / (area1 + area2). Guard division by zero.
    if (area1 + area2) == 0:
        return 0.0

    dice = (2.0 * intersection_area) / (area1 + area2)
    return dice


def load_and_convert_csv(csv_path):
    # Load a CSV and convert serialized bbox strings into Python lists/tuples
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        # Return empty DataFrame if file has no content
        print(f"File {csv_path} is empty.")
        return pd.DataFrame()

    # Convert string representations of lists/tuples into actual Python objects
    df['true_bbox'] = df['true_bbox'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['predicted_bbox'] = df['predicted_bbox'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    return df


def calculate_tp_fp_fn_tn(df, iou_threshold=0.95, conf_threshold=0.5):
    """
    Vectorized, per-row evaluation:
    - TP: true present AND pred present (confidence >= conf_threshold) AND same class AND IoU >= iou_threshold
    - FP: pred present (confidence >= conf_threshold) AND NOT TP
    - FN: true present AND NOT TP
    - TN: neither true nor pred present (pred also must be below conf_threshold)
    Returns per-class TP/FP/FN dicts, a global TN count under key 'background', and iou lists per class for TP matches.
    """
    # Ensure working on a copy to avoid modifying caller dataframe
    df_work = df.copy()

    # Normalize None-like values to pandas NaN for straightforward masks
    df_work['true_class'] = df_work['true_class'].where(pd.notna(df_work['true_class']), np.nan)
    df_work['predicted_class'] = df_work['predicted_class'].where(pd.notna(df_work['predicted_class']), np.nan)
    df_work['predicted_bbox'] = df_work['predicted_bbox'].where(pd.notna(df_work['predicted_bbox']), None)
    df_work['true_bbox'] = df_work['true_bbox'].where(pd.notna(df_work['true_bbox']), None)

    # Boolean masks
    pred_mask = (df_work['confidence'] >= conf_threshold) & df_work['predicted_bbox'].notna()
    true_mask = df_work['true_class'].notna()

    # Rows where both a true and a (confident) prediction exist
    both_mask = pred_mask & true_mask

    # Compute IoUs only for rows where both exist
    iou_values = np.zeros(len(df_work), dtype=float)
    if both_mask.any():
        # faster row-wise computation with list comprehension for only necessary rows
        idxs = np.flatnonzero(both_mask.values)
        ious_list = [
            compute_iou(df_work.iloc[i]['true_bbox'], df_work.iloc[i]['predicted_bbox'])
            for i in idxs
        ]
        iou_values[idxs] = ious_list

    # Attach iou_values into df_work for easy masking
    df_work = df_work.assign(_iou_calc=iou_values)

    # True Positive mask: both present AND class match AND iou >= threshold
    tp_mask = both_mask & (df_work['true_class'] == df_work['predicted_class']) & (df_work['_iou_calc'] >= iou_threshold)

    # False Positive mask: prediction present (above conf) but not TP
    fp_mask = pred_mask & (~tp_mask)

    # False Negative mask: true present but not TP
    fn_mask = true_mask & (~tp_mask)

    # True Negative mask: no true and no confident prediction
    tn_mask = (~true_mask) & (~pred_mask)

    # Aggregate counts per class
    true_positives = defaultdict(int)
    false_positives = defaultdict(int)
    false_negatives = defaultdict(int)
    iou_scores = defaultdict(list)

    # Collect classes from both true and predicted columns (drop NaN)
    classes = set(pd.concat([df_work['true_class'].dropna().astype(int), df_work['predicted_class'].dropna().astype(int)]).unique())

    for cls in classes:
        cls = int(cls)
        true_positives[cls] = int(np.sum(tp_mask & (df_work['true_class'] == cls)))
        false_positives[cls] = int(np.sum(fp_mask & (df_work['predicted_class'] == cls)))
        false_negatives[cls] = int(np.sum(fn_mask & (df_work['true_class'] == cls)))
        # iou list only for TP rows of that class
        if cls in true_positives and true_positives[cls] > 0:
            iou_scores[cls] = df_work.loc[tp_mask & (df_work['true_class'] == cls), '_iou_calc'].tolist()
        else:
            iou_scores[cls] = []

    # True negatives treated as global 'background'
    true_negatives = {'background': int(tn_mask.sum())}

    return true_positives, false_positives, false_negatives, true_negatives, iou_scores

def calculate_all_metrics(df, iou_threshold=0.95, conf_threshold=0.5):
    """
    Build per-class metrics and overall totals using vectorized counts returned by calculate_tp_fp_fn_tn.
    """
    tp, fp, fn, tn, iou_scores = calculate_tp_fp_fn_tn(df, iou_threshold=iou_threshold, conf_threshold=conf_threshold)
    metrics = {}

    # Gather all classes seen
    all_classes = set(tp.keys()).union(fp.keys()).union(fn.keys())

    for class_id in sorted(all_classes):
        tp_c = tp.get(class_id, 0)
        fp_c = fp.get(class_id, 0)
        fn_c = fn.get(class_id, 0)
        tn_c = tn.get('background', 0)

        precision = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        recall = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp_c + tn_c) / (tp_c + tn_c + fp_c + fn_c) if (tp_c + tn_c + fp_c + fn_c) > 0 else 0.0
        iou_mean = float(np.mean(iou_scores.get(class_id, []))) if iou_scores.get(class_id, []) else 0.0

        metrics[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'iou_mean': iou_mean,
            'tp': tp_c,
            'fp': fp_c,
            'fn': fn_c,
            'tn': tn_c,
        }

    # Totals across all classes
    tp_total = sum(tp.values())
    fp_total = sum(fp.values())
    fn_total = sum(fn.values())
    tn_total = tn.get('background', 0)

    precision_total = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall_total = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1_total = 2 * (precision_total * recall_total) / (precision_total + recall_total) if (precision_total + recall_total) > 0 else 0.0
    accuracy_total = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total) if (tp_total + tn_total + fp_total + fn_total) > 0 else 0.0
    all_ious = [iou for lst in iou_scores.values() for iou in lst]
    iou_mean_total = float(np.mean(all_ious)) if all_ious else 0.0

    metrics['total'] = {
        'precision': precision_total,
        'recall': recall_total,
        'f1': f1_total,
        'accuracy': accuracy_total,
        'iou_mean': iou_mean_total,
        'tp': tp_total,
        'fp': fp_total,
        'fn': fn_total,
        'tn': tn_total,
    }

    return metrics


def display_metrics(metrics):
    # Pretty-print metrics per-class and the overall totals
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
    # Load multiple CSV files and concatenate them into a single DataFrame
    data_frames = []
    for path in file_paths:
        df = pd.read_csv(path)
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)


def extract_model_name_and_conf(dir_name):
    # Extract model name and confidence threshold encoded in directory name like "model_conf_50"
    match = re.search(r'(.+)_conf_(\d+\.?\d*)', dir_name)
    if match:
        model_name = match.group(1)
        # Directory stores confidence as percentage; convert to fraction
        conf_threshold = float(match.group(2)) / 100
        return model_name, conf_threshold
    return None, None


def load_and_evaluate(file_path, conf_threshold, iou_threshold):
    # Load CSV and compute summary metrics for given confidence and IoU thresholds
    # Expects columns: confidence, iou, dice, is_FP, is_FN, is_TN
    try:
        data = pd.read_csv(file_path)
        # Filter rows that satisfy both confidence and IoU thresholds (for counting detections)
        filtered_data = data[(data['confidence'] >= conf_threshold) & (data['iou'] >= iou_threshold)]

        # Compute counts using boolean flags in the CSV (assumes those columns are present)
        tp = len(filtered_data[filtered_data['is_FP'] == False])
        fp = len(data[data['is_FP'] == True])
        fn = len(data[data['is_FN'] == True])
        tn = len(data[data['is_TN'] == True])

        # Compute metrics with guards against division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        avg_iou = filtered_data['iou'].mean() if len(filtered_data) > 0 else 0
        avg_dice = filtered_data['dice'].mean() if len(filtered_data) > 0 else 0

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
        # Print error and return None to skip problematic files
        print(f"Error processing {file_path}: {e}")
        return None


def evaluate_all_files(root_dir, iou_thresholds, output_file):
    # Walk through subdirectories of `root_dir`, each named like 'model_conf_XX', and evaluate CSV files
    results = []
    total_examples = None

    # Determine total_examples from first found CSV to use as a reference count
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

    # Iterate over model directories and their CSV files, filtering by IoU thresholds of interest
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        model_name, conf_threshold_dir = extract_model_name_and_conf(dir_name)
        if model_name is None:
            continue

        for file_name in os.listdir(dir_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(dir_path, file_name)
                # Extract IoU encoded in filename like 'results_iou_50.csv'
                iou_match = re.search(r'iou_(\d+\.?\d*)', file_name)
                if iou_match:
                    iou_threshold = float(iou_match.group(1)) / 100
                    if iou_threshold in iou_thresholds:
                        result = load_and_evaluate(file_path, conf_threshold_dir, iou_threshold)
                        if result:
                            results.append(result)

    # Save aggregated results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Les résultats ont été enregistrés dans {output_file}")
    return results_df


def plot_metrics_from_dataframe(data, output_plot_dir):
    # Plot metrics versus confidence thresholds for each model and IoU combination versus confidence threshold
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
    pass