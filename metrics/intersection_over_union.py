from tensorflow.keras import backend


def iou(y_true, y_pred, label=1):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = backend.cast(backend.equal(backend.argmax(y_true), label), backend.floatx())
    y_pred = backend.cast(backend.equal(backend.argmax(y_pred), label), backend.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = backend.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = backend.sum(y_true) + backend.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return backend.switch(backend.equal(union, 0), 1.0, intersection / union)
