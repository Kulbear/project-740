import numpy as np
from .base_metric import BaseMetric
from sklearn.metrics import confusion_matrix, cohen_kappa_score

__all__ = [
    'FBetaMetric',
    'AccuracyMetric',
    'RecallMetric',
    'PrecisionMetric',
    'QuadraticKappa',
    'CohenKappa']


class CohenKappa(BaseMetric):
    def __init__(self, metadata={}):
        self.weights = metadata.get('weights', 'quadratic')

    def __call__(self, prediction, target):
        """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
        at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values
        of adoption rating."""

        actuals = target
        preds = prediction.argmax(axis=-1)
        return cohen_kappa_score(preds, actuals, weights=self.weights)


class QuadraticKappa(BaseMetric):
    def __init__(self, metadata={}):
        pass
    
    def __call__(self, prediction, target):
        """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
        at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values 
        of adoption rating."""

        actuals = target
        preds = prediction.argmax(axis=-1)
        N = preds.shape[0]
        w = np.zeros((N, N))
        O = confusion_matrix(actuals, preds)
        for i in range(len(w)):
            for j in range(len(w)):
                w[i][j] = float(((i - j) ** 2) / (N - 1) ** 2)

        act_hist = np.zeros([N])
        for item in actuals:
            act_hist[item] += 1

        pred_hist = np.zeros([N])
        for item in preds:
            pred_hist[item] += 1

        E = np.outer(act_hist, pred_hist)
        E = E / E.sum()
        O = O / O.sum()

        num = 0
        den = 0
        for i in range(len(w)):
            for j in range(len(w)):
                num += w[i][j] * O[i][j]
                den += w[i][j] * E[i][j]
        return (1 - (num / den))


class FBetaMetric(BaseMetric):
    """Compute the F-beta score (from sklearn doc).

    The F-beta score is the weighted harmonic mean of precision and recall,
    reaching its optimal value at 1 and its worst value at 0.

    The beta parameter determines the weight of recall in the combined score.
    beta < 1 lends more weight to precision, while beta > 1 favors recall
    (beta -> 0 considers only precision, beta -> inf only recall).
    """

    def __init__(self, metadata={}):
        super().__init__(metadata=metadata)
        self.eps = metadata.get('eps', 1e-8)
        self.beta = metadata.get('beta', 1)
        self.threshold = metadata.get('threshold', 0.5)

    def __call__(self, prediction, target):
        prediction = np.array((prediction > self.threshold), dtype=np.int8)

        assert target.shape[0] == target.shape[0]
        tp = np.sum((target == 1) & (prediction == 1))
        fp = np.sum((target == 0) & (prediction == 1))
        fn = np.sum((target == 1) & (prediction == 0))
        p = tp / (tp + fp + self.eps)
        r = tp / (tp + fn + self.eps)
        fbeta = (1 + self.beta ** 2) * p * r / (p * self.beta ** 2 + r + self.eps)

        return fbeta


class AccuracyMetric(BaseMetric):
    """Accuracy classification score."""

    def __init__(self, metadata={}):
        super().__init__(metadata=metadata)
        self.threshold = metadata.get('threshold', 0.5)
        self.is_softmax = metadata.get('is_softmax', False)

    def __call__(self, prediction, target):
        batch_size = float(target.shape[0])
        if not self.is_softmax:
            prediction = np.array((prediction > self.threshold),
                                  dtype=np.uint8)
            pred_true = target == prediction
        else:
            pred_true = (prediction.argmax(axis=1) == target).sum()

        return pred_true / batch_size


class RecallMetric(BaseMetric):
    """Compute the recall (from sklearn doc).

    The recall is the ratio tp / (tp + fn)
    where tp is the number of true positives and fn the number of false negatives.

    The recall is intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.
    """

    def __init__(self, metadata={}):
        super().__init__(metadata=metadata)
        self.eps = metadata.get('eps', 1e-8)
        self.threshold = metadata.get('threshold', 0.5)

    def __call__(self, prediction, target):
        prediction = np.array((prediction > self.threshold), dtype=np.int8)

        assert target.shape[0] == target.shape[0]
        tp = np.sum((target == 1) & (prediction == 1))
        fn = np.sum((target == 1) & (prediction == 0))
        recall = tp / (tp + fn + self.eps)

        return recall


class PrecisionMetric(BaseMetric):
    """Compute the precision (from sklearn doc).

    The precision is the ratio tp / (tp + fp)
    where tp is the number of true positives and fp the number of false positives.

    The precision is intuitively the ability of the classifier
    not to label as positive a sample that is negative.

    The best value is 1 and the worst value is 0.
    """

    def __init__(self, metadata={}):
        super().__init__(metadata=metadata)
        self.eps = metadata.get('eps', 1e-8)
        self.threshold = metadata.get('threshold', 0.5)

    def __call__(self, prediction, target):
        prediction = np.array((prediction > self.threshold), dtype=np.int8)

        assert target.shape[0] == target.shape[0]
        tp = np.sum((target == 1) & (prediction == 1))
        fp = np.sum((target == 0) & (prediction == 1))
        prec = tp / (tp + fp + self.eps)

        return prec
