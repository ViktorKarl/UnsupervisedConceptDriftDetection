from collections import deque
import numpy as np
from scipy.stats import gaussian_kde
from scipy.special import rel_entr
from scipy.stats import entropy
from .base import UnsupervisedDriftDetector


class KullbackLeiblerDistanceDetector(UnsupervisedDriftDetector):
    """
    Drift detection using Kullback-Leibler divergence between probability distributions.
    """

    def __init__(self, window_len: int, step_size: int = 10 ,threshold: float = 0.1):
        super().__init__()
        self.window_len = window_len
        self.threshold = threshold
        self.data_window = deque(maxlen=window_len)
        self.step_size = step_size
        self.reference_window = deque(maxlen=window_len//2) 
        self.recent_window = deque(maxlen=window_len//2)

    def update(self, buffer: list, clear_window: bool) -> bool:
        """
        Update the detector with the most recent observation and determine if a drift occurred.

        :param features: the features
        :returns: True if a drift was detected else False
        """
        self.data_window.extend(buffer)
        if len(self.data_window) == self.window_len:
            data = np.array(self.data_window)
            for i in range(data.shape[1]):
                reference_data, recent_data = self._get_samples(i)
                all_data = np.concatenate((reference_data, recent_data)).flatten()
                sample_points = np.linspace(all_data.min(), all_data.max(), self.window_len//2)
                p = self._estimate_pdf(reference_data, sample_points)
                q = self._estimate_pdf(recent_data, sample_points)
                kl_div = self._kullback_leibler_divergence(p, q)
                kl_div = self._kullback_leibler_divergence(p, q)
                if kl_div > self.threshold:
                    if clear_window:
                        self.data_window.clear()
                    return True
        return False
    
    # def update(self, features: dict) -> bool:
    #     """
    #     Update the detector with the most recent observation and determine if a drift occurred.

    #     :param features: the features
    #     :returns: True if a drift was detected else False
    #     """
    #     features = np.fromiter(features.values(), dtype=float)
    #     self.data_window.append(features)
    #     if len(self.data_window) == self.data_window.maxlen:
    #         data = np.array(self.data_window)
    #         for i in range(data.shape[1]):
    #             sample_one, sample_two = self._get_samples(i)
    #             p = self._estimate_pdf(sample_one)
    #             q = self._estimate_pdf(sample_two)
    #             kl_div = self._kullback_leibler_divergence(p, q)
    #             if kl_div > self.threshold:
    #                 self.reset()
    #                 return True
    #     return False

    def reset(self):
        """
        Reset the drift detector by clearing the data window.
        """
        self.data_window = deque(maxlen=self.window_len)

    def _get_samples(self, feature_index):
        data = np.array(self.data_window)
        data_slice = data[:, feature_index]
        refrence_data = data_slice[:self.window_len//2]
        recent_data = data_slice[self.window_len//2:]
        return refrence_data, recent_data

    def _estimate_pdf(self, data, sample_points):
        data_flat = data.flatten()
        kde = gaussian_kde(data_flat)
        pdf = kde(sample_points)
        pdf /= np.sum(pdf)  # normalise
        return pdf

    def _kullback_leibler_divergence(self, p, q):
        # Avoid division by zero
        p = np.where(p == 0, 1e-10, p)
        q = np.where(q == 0, 1e-10, q)
        kl_div = np.sum(rel_entr(p, q))
        return kl_div
