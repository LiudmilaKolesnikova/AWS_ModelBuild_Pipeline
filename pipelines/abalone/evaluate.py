"""Evaluation script for measuring F1-score."""
import json
import logging
import pathlib
import pickle
import tarfile
import os

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())



if __name__ == "__main__":
    
    logger.debug("Starting evaluation.")
    y_batch_transform_path = "/opt/ml/processing/input/ground_truth_with_predictions/test.csv.out"
    y_batch_transform = pd.read_csv(y_batch_transform_path, header=None)
    
    logger.debug("Reading transform data.")
    y_true = y_batch_transform[0].to_numpy()
    y_pred = y_batch_transform[1].to_numpy()
    
    logger.debug("Calculating f1 score.")
    f1 = f1_score(y_true, y_pred, average="weighted")
    
    report_dict = {
        "classification_metrics": {
            "weighted_f1": {
                "value": f1
            },
        },
    }
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Writing out evaluation report with f1: %f", f1)
    evaluation_path = os.path.join(output_dir, "evaluation.json")
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
