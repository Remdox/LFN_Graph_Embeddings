import sklearn.metrics as metrics


def evaluate_AUROC(y_true, y_pred):
    """
    Evaluare the Area Under the Receiver Operating Characteristic Curve (AUROC).

    Parameters:
    - y_true: true binary labels,  
    - y_pred: target scores.

    Returns:
    - AUROC value.
    """
    
    return metrics.roc_auc_score(y_true, y_pred)


def evaluate_AUPR(y_true, y_pred):
    """
    Evaluare the Area Under the Precision-Recall curve (AUPR).

    Parameters:
    - y_true: true binary labels,  
    - y_pred: target scores.

    Returns:
    - AUPR value.
    """

    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    return metrics.auc(recall, precision)

