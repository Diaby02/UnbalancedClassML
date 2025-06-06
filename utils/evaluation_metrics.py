from sklearn.metrics import accuracy_score

def balanced_accuracy(y_true,y_pred):
    """
    Balanced Classification Rate
    """
    # number of classes
    n_classes = len(set(y_true))
    # initialize the BCR
    bcr = 0
    # for each class
    for i in range(n_classes):
        # get the indices of the class i
        idx = y_true == i
        # compute the accuracy of the class i
        acc = accuracy_score(y_true[idx], y_pred[idx])
        # update the BCR
        bcr += acc
    # return the BCR
    return bcr / n_classes


def accuracy(y_true, y_pred):

    return accuracy_score(y_true, y_pred)