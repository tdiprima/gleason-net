"""
Evaluation functions for Prostate Cancer Gleason Grading
"""
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def evaluate(model, loader, device):
    """
    Evaluate the model on a dataset using sklearn metrics.

    Returns:
        accuracy, precision, recall, f1, confusion_matrix
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            if data is None:
                continue

            data = data.to(device)
            target = target.to(device)

            output = model(data)
            preds = output.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, f1, cm
