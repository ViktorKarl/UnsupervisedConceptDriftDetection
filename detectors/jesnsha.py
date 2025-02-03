from collections import deque
import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from .base import UnsupervisedDriftDetector


class JensenShannonDistanceDetector(UnsupervisedDriftDetector):
    """
    Drift detection using Jensen-Shannon divergence between probability distributions.
    """

    def __init__(self, window_len: int, threshold: float = 0.1):
        super().__init__()
        self.window_len = window_len
        self.threshold = threshold
        self.data_window = deque(maxlen=window_len)


    def update_new(self, buffer: list) -> bool:
        """
        Update the detector with the most recent observation and determine if a drift occurred.

        :param features: the features
        :returns: True if a drift was detected else False
        """
        state = False
        self.data_window.extend(buffer)
        if len(self.data_window) == self.window_len:
            data = np.array(self.data_window)
            for i in range(data.shape[1]):
                refrence_data, recent_data = self._get_samples(i)
                p = self._estimate_pdf(refrence_data)
                q = self._estimate_pdf(recent_data)
                js_dist = self._jensen_shannon_distance(p, q)
                if js_dist > self.threshold:
                    state = True
            if state:
                    self.data_window.clear()
                    return True
        else:
            return False

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
        self.data_window = deque(maxlen=2 * self.window_len)

    def _get_samples(self, feature_index):
        data = np.array(self.data_window)
        data_slice = data[:, feature_index]
        refrence_data = data_slice[:self.window_len//2]
        recent_data = data_slice[self.window_len//2:]
        return refrence_data, recent_data

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