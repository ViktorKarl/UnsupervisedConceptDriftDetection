# File: optimize_optuna/optuna_optimizer.py
import optuna

class OptunaOptimizer:
    def __init__(self, objective, parameters, n_trials=50, storage="sqlite:///optuna.db", study_name="optuna_study"):
        """
        :param objective: A callable that takes a configuration dict and returns a fitness value.
        :param parameters: A list of Parameter objects defining the hyperparameter search space.
        :param n_trials: The number of trials to run.
        :param storage: Database storage for Optuna (default is SQLite).
        :param study_name: Name of the study (for easier identification in the dashboard).
        """
        self.objective = objective
        self.parameters = parameters
        self.n_trials = n_trials
        self.storage = storage
        self.study = optuna.create_study(
            direction="maximize",
            storage=self.storage,
            study_name=study_name,
            load_if_exists=True
        )

    def sample_parameters(self, trial):
        """Sample a configuration based on the parameter definitions."""
        config = {}
        for param in self.parameters:
            if len(param.values) == 2 and all(isinstance(v, (int, float)) for v in param.values):
                lower, upper = param.values
                if isinstance(lower, int) and isinstance(upper, int):
                    if param.name in ['window_len', 'step_size']:
                        lower = lower if lower % 2 == 0 else lower + 1  # Ensure lower bound is even
                        upper = upper if upper % 2 == 0 else upper - 1  # Ensure upper bound is even
                        config[param.name] = trial.suggest_int(param.name, lower, upper, step=2)
                    else:
                        config[param.name] = trial.suggest_int(param.name, lower, upper)
                else:
                    config[param.name] = trial.suggest_float(param.name, lower, upper)
            else:
                config[param.name] = trial.suggest_categorical(param.name, param.values)
        return config

    def optimize(self):
        """Run the Optuna study and return the best configuration and its fitness."""
        def objective_wrapper(trial):
            config = self.sample_parameters(trial)
            return self.objective(config, trial)

        self.study.optimize(objective_wrapper, n_trials=self.n_trials)
        return self.study.best_params, self.study.best_value
