from collections import deque
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from .base import UnsupervisedDriftDetector


class JensenShannonDistanceDetector(UnsupervisedDriftDetector):
    """
    Drift detection using Jensen-Shannon divergence between probability distributions.
    """

    def __init__(self, n_samples: int, threshold: float = 0.1, seed=None):
        super().__init__(seed)
        self.n_samples = n_samples
        self.threshold = threshold
        self.data_window = deque(maxlen=2 * n_samples)

    def update(self, features: dict) -> bool:
        """
        Update the detector with the most recent observation and determine if a drift occurred.

        :param features: the features
        :returns: True if a drift was detected else False
        """
        features = np.fromiter(features.values(), dtype=float)
        self.data_window.append(features)
        if len(self.data_window) == self.data_window.maxlen:
            data = np.array(self.data_window)
            for i in range(data.shape[1]):
                sample_one, sample_two = self._get_samples(i)
                p = self._estimate_pdf(sample_one)
                q = self._estimate_pdf(sample_two)
                js_dist = self._jensen_shannon_distance(p, q)
                if js_dist > self.threshold:
                    self.reset()
                    return True
        return False

    def reset(self):
        """
        Reset the drift detector by clearing the data window.
        """
        self.data_window = deque(maxlen=2 * self.n_samples)

    def _get_samples(self, feature_index):
        data = np.array(self.data_window)
        data_slice = data[:, feature_index]
        sample_one = data_slice[:self.n_samples]
        sample_two = data_slice[self.n_samples:]
        return sample_one, sample_two

    def _estimate_pdf(self, data):
        data_flat = data.flatten()
        sample_points = np.linspace(np.min(data_flat), np.max(data_flat), 1000)
        kde = gaussian_kde(data_flat)
        pdf = kde(sample_points)
        # Normalize pdf
        pdf /= np.sum(pdf)
        return pdf

    def _jensen_shannon_distance(self, p, q):
        dist = jensenshannon(p, q)
        return dist