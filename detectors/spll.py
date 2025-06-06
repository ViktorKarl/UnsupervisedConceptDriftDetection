from collections import deque
from typing import Optional
from itertools import islice

import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import chi2
from sklearn.cluster import KMeans

from .base import UnsupervisedDriftDetector


class SemiParametricLogLikelihood(UnsupervisedDriftDetector):
    """
    SemiParametricLogLikelihood (SPLL) detects concept drifts by calculating the log-likelihood that both the recent
    data and the reference data stem from the same distributions. A Gaussian mixture models based on k-means clustering
    is used to estimate the density of both data windows. In order to improve the stability of the likelihood
    calculation, the covariance matrix is calculated over a full data window instead of calculating a covariance matrix
    per component. Incoming data are assigned to the closest component of the Gaussian mixture model as determined with
    the Mahalanobis distance.

    Source: Kuncheva, L. (2013). Change detection in streaming multivariate data using likelihood detectors. IEEE
        Transactions on Knowledge and Data Engineering.
    """

    def __init__(
        self,
        window_len: int,
        n_clusters: int,
        threshold: float,
        step_size: int = 10
    ):
        """
        Init a new SPLL drift detector.

        :param n_samples: the size of the reference and recent data windows
        :param n_clusters: the number of clusters created by the kmeans algorithm
        :param threshold: the threshold for a drift detection
        """
        super().__init__()
        self.window_len = window_len
        self.data_window = deque(maxlen=window_len)
        self.recent_window = deque(maxlen=window_len//2)
        self.reference_window = deque(maxlen=window_len//2)
        self.step_size = step_size
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed)
        self.threshold = threshold

    def update(self, buffer: list, clear_window: bool) -> bool:
        """
        Update the detector with the most recent features.

        :param features: the features
        :return: True if a drift was detected, else False
        """
        self.data_window.extend(buffer)
        self.reference_window.clear()
        self.recent_window.clear()

        self.reference_window.extend(islice(self.data_window,0,self.window_len//2))
        self.recent_window.extend(islice(self.data_window, self.window_len//2, None))

        if len(self.reference_window) == self.window_len//2 and len(self.recent_window) == self.window_len//2:
            drift = self._detect_drift()
            if drift:
                if clear_window:
                    self.data_window.clear()
                return True
        return False
        
    # def update(self, features: dict) -> bool:
    #         """
    #         Update the detector with the most recent features.
    
    #         :param features: the features
    #         :return: True if a drift was detected, else False
    #         """
    #         features = np.fromiter(features.values(), dtype=float)
    #         if len(self.recent_data) == self.n_samples:
    #             self.reference_data.append(self.recent_data[0])
    #         self.recent_data.append(features)
    #         if len(self.reference_data) == self.n_samples and len(self.recent_data) == self.n_samples:
    #             drift = self._detect_drift()
    #             if drift:
    #                 self.reset()
    #                 return True
    #         return False

    def _detect_drift(self) -> bool:
        """
        Detect if a concept drift occurred.

        :return: True if a drift occurred, else False
        """
        reference_data = np.array(self.reference_window)
        self.kmeans.fit(reference_data)
        # Kuncheva suggests computing the covariance matrix over the entire dataset instead of over components for
        # improved stability
        inverse_covariance_matrix = np.linalg.pinv(np.cov(reference_data.T))
        closest_centroids = self._calculate_closest_centroids(inverse_covariance_matrix)
        spll = self._calculate_spll(inverse_covariance_matrix, closest_centroids)
        probability = np.exp(-spll)
        # Kuncheva suggests using the dimensionality of data as degrees of freedom in the chi2 test
        degrees_of_freedom = len(self.recent_window[0])
        quantile = chi2.ppf(probability, degrees_of_freedom)
        drift = quantile < self.threshold
        return drift

    def _calculate_closest_centroids(
        self,
        inverse_covariance_matrix: np.array,
    ) -> np.array:
        """
        Calculate the closest centroid for each recent sample based on the Mahalanobis distance.

        :param inverse_covariance_matrix: the inverse covariance matrix used for the Mahalanobis distance
        :return: the closest centroids
        """
        distances_to_centers = distance.cdist(
            self.recent_window,
            self.kmeans.cluster_centers_,
            metric="mahalanobis",
            VI=inverse_covariance_matrix,
        )
        centroid_indices = np.argmin(distances_to_centers, axis=1)
        centroids = [
            self.kmeans.cluster_centers_[closest] for closest in centroid_indices
        ]
        return np.array(centroids)

    def _calculate_spll(
        self,
        inverse_covariance_matrix: np.array,
        closest_centroids: np.array,
    ) -> float:
        """
        Calculate the log likelihood of the most recent data coming from the same distribution as the reference data,
        which was used to construct the Gaussian mixture model.

        :param inverse_covariance_matrix: the inverse covariance matrix used to calculate the likelihood
        :param closest_centroids: the closest centroid of each recent sample
        :return: the log likelihood
        """
        recent_data = np.array(self.recent_window)
        centered = recent_data - closest_centroids
        likelihoods = []
        for i in range(self.window_len//2):
            normalized_x = np.matmul(centered[i], inverse_covariance_matrix)
            likelihood = np.matmul(normalized_x, centered[i])
            likelihoods.append(likelihood)
        spll = np.sum(likelihoods) / self.window_len
        return spll

    def reset(self):
        """
        Reset the drift detector by clearing the recent data and resetting the KMeans.
        """
        self.reference_window = self.recent_window
        self.recent_window = deque(maxlen=self.window_len)
