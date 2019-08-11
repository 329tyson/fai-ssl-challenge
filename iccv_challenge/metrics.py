from sklearn.metrics import average_precision_score


def mean_average_precision(prediction, labels, num_class=20):
    '''
    This functions is designed to evaluate performance of multi-class \
        multi-label classification task
    Args:
        prediction: (dataset_size, num_class) outputs of model
        label: (dataset_size, num_class) binary answer labels
    '''
    mAP = 0
    for i in range(num_class):
        pred = prediction[:, i]
        label = labels[:, i]
        mAP += average_precision_score(label, pred)

    return mAP / num_class * 100
