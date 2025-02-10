

from run_detectors.config_generator import ConfigGenerator
from optimization.logger import ExperimentLogger
from run_detectors.parameter import Parameter 
from typing import List, Optional

import pandas as pd
import numpy as np
from collections import deque



class detector_runner:
        def __init__(
            self,
            base_model: callable,
            parameters: List[Parameter],
        ):
            """
            Init a new ModelOptimizer.
            :param base_model: a callable of the detector under test
            :param parameters: the configuration parameters
            :param n_runs: the number of test runs for each configuration
            :param seeds: the seeds or None
            """
            self.base_model = base_model
            self.configs = ConfigGenerator(parameters)
            self.classifiers = None

        def initialize_logger(self,stream,experiment_name,config,verbose=False): 
                self.logger = ExperimentLogger( stream=stream,
                model=self.base_model.__name__,
                experiment_name=experiment_name,
                config_keys=self.configs.get_parameter_names(),
                )
                print(f"{self.logger.model}: {config}") if verbose else None

        def _model_generator(self):
            """
            A generator that yields initialized models using configurations provided by the ConfigGenerator.

            :return: the initialized models
            """
            for config in self.configs:
                yield self.base_model(**config), config

        def run(self,stream):
            """
            run datastream through model, handeling windowing 

            :param stream: the data stream
            :param experiment_name: the name of the experiment
            :param n_training_samples: the number of training samples
            """
            buffer = []
            for model, config in self._model_generator():
                drifts = []
                for i, (sample, lable) in enumerate(stream):
                    if model.window_len == len(model.data_window):
                        buffer.append(np.fromiter(sample.values(), dtype=float))
                        if len(buffer) == model.step_size:
                            if model.update(buffer):
                                drifts.append(i)
                            buffer.clear()
                    else:    
                        model.data_window.append(np.fromiter(sample.values(), dtype=float))
                        if len(buffer) == 0 and len(model.data_window) == model.window_len:
                            if model.update(buffer):
                                drifts.append(i)
            return drifts