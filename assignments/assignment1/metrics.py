import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    """
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    zipped = np.dstack((prediction, ground_truth))
    for p in np.rollaxis(zipped, 1):
        pair = p[0]
        if pair[0]:
            if pair[1]:
                tp += 1
            else:
                tn += 1
        else:
            if pair[1]:
                fn += 1
            else:
                fp += 1
    # np.apply_along_axis(resolve_case, axis=2, arr=zipped)

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    correct = 0

    zipped = np.dstack((prediction, ground_truth))
    for p in np.rollaxis(zipped, 1):
        pair = p[0]
        if pair[0] == pair[1]:
            correct += 1

    return correct / prediction.shape[0]
