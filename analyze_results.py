import os

import tools as tools
import configuration as cfg


if __name__ == "__main__":
    iou_thresholds = [0.5, 0.75, 0.95]
    csv_file = os.path.join(cfg.RESULT_DIR, 'compared_results.csv')
    plot_dir = os.path.join(cfg.RESULT_DIR, 'plots')

    results_df = tools.evaluate_all_files(cfg.RESULT_DIR, iou_thresholds, csv_file)
    tools.plot_metrics_from_dataframe(results_df, output_plot_dir=plot_dir)