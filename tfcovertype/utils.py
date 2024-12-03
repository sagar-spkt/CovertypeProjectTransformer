from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    """
    Compute evaluation metrics for model predictions.

    This function calculates the accuracy, precision, recall, and F1-score
    for the given predictions and true labels.

    Parameters
    ----------
    pred : transformers.EvalPrediction
        An object containing the predictions and true labels. It should have
        the following attributes:
        - `predictions`: ndarray of shape (n_samples, n_classes)
          The raw model predictions (logits).
        - `label_ids`: ndarray of shape (n_samples,)
          The true labels.

    Returns
    -------
    dict
        A dictionary containing the following evaluation metrics:
        - 'accuracy': float
            The accuracy of the predictions.
        - 'precision': float
            The precision of the predictions, computed as the macro average.
        - 'recall': float
            The recall of the predictions, computed as the macro average.
        - 'f1': float
            The F1-score of the predictions, computed as the macro average.

    Examples
    --------
    >>> from transformers import EvalPrediction
    >>> import numpy as np
    >>> pred = EvalPrediction(predictions=np.array([[0.1, 0.9], [0.8, 0.2]]), label_ids=np.array([1, 0]))
    >>> metrics = compute_metrics(pred)
    >>> print(metrics)
    {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    """
    # Get the predicted class labels by taking the argmax over the last axis
    predictions = pred.predictions.argmax(axis=-1)

    # Extract the true labels
    labels = pred.label_ids

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)

    # Calculate precision, recall, and F1-score (macro average)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')

    # Return metrics as a dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
