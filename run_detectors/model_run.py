from metrics.metrics import get_metrics
from optimization.classifiers import Classifiers
from optimization.config_generator import ConfigGenerator
from optimization.logger import ExperimentLogger
from optimization.parameter import Parameter 
from typing import List, Optional

import pandas as pd
import numpy as np
from collections import deque



class run_model:
        def __init__(
            self,
            base_model: callable,
            parameters: List[Parameter],
            seeds: Optional[List[int]] = None
        ):
            """
            Init a new ModelOptimizer.
            :param base_model: a callable of the detector under test
            :param parameters: the configuration parameters
            :param n_runs: the number of test runs for each configuration
            :param seeds: the seeds or None
            """
            self.base_model = base_model
            self.configs = ConfigGenerator(parameters, seeds=seeds)
            self.classifiers = None

        def initialize_logger(self,stream,experiment_name): 
                self.logger = ExperimentLogger( stream=stream,
                model=self.base_model.__name__,
                experiment_name=experiment_name,
                config_keys=self.configs.get_parameter_names(),
                )

        def _model_generator(self):
            """
            A generator that yields initialized models using configurations provided by the ConfigGenerator.

            :return: the initialized models
            """
            for config in self.configs:
                yield self.base_model(**config), config

        def run_model(self,stream, step_size = 1, verbose=True):
            """
            Optimize the model on the given data stream and log the results using the ExperimentLogger.

            :param stream: the data stream
            :param experiment_name: the name of the experiment
            :param n_training_samples: the number of training samples
            """
            buffer = []
            for model, config in self._model_generator():
                print(f"{self.logger.model}: {config}") if verbose else None
                drifts = []
                for i, (sample, lable) in enumerate(stream):
                    if len(model.data_window) 
                    buffer.append(np.fromiter(sample.values(), dtype=float))
                    if len(buffer) == step_size:
                        if model.update(buffer):
                            drifts.append(i)
                            buffer.clear()
            return drifts