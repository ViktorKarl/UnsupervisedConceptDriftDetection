import itertools
import time
from typing import List, Optional

from run_detectors.parameter import Parameter


class ConfigGenerator:
    """
    A generator providing configurations based on a list of parameter values.
    """

    def __init__(self, parameters: List[Parameter] ):
        """
        Init a new config generator with the given list of parameters and seeds (optional).

        :param parameters: the list of parameters
        :param seeds: the seeds or None
        """
        self.parameters = sorted(parameters, key=lambda p: p.name)

    def get_parameter_names(self):
        """
        Get the name of each parameter.

        :return: the names
        """
        names = [parameter.name for parameter in self.parameters]
        return names

    def __iter__(self):
        """
        Creates all combinations of configurations from the parameters and yields them. If no seeds were provided,
        the current UNIX time is used as seed instead.

        :return: the configurations
        """
        all_parameters = [list(parameter) for parameter in self.parameters]
        for i, combination in enumerate(itertools.product(*all_parameters)):
            config = {
                parameter.name: combination[j]
                for j, parameter in enumerate(self.parameters)
            }
            yield config
