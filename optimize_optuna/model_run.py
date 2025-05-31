from run_detectors.parameter import Parameter
import numpy as np
import optuna
from .classifiers import Classifiers
from metrics.metrics import get_metrics
from optimize_optuna.optuna_optimizer import OptunaOptimizer  # our Optuna wrapper from before
from optimize_optuna.evaluation_metrics import accuracy_metric, confusion_matrix, f1_metric

def fill_predictions(predictions, step_size, recent_window_maxlen):
    output = []
    for value, flag in predictions:
        # Generate the sequence: [value - step_size + 1, ..., value] with the same flag
        if step_size < recent_window_maxlen:
            for num in range(value - step_size + 1, value + 1):
                output.append([num, flag])

        elif recent_window_maxlen <= step_size:
            for num in range(value - recent_window_maxlen + 1, value + 1):
                output.append([num, flag])
    return output


def refrence_cutoff(cut_off, refrence_window_maxlen):
    fp = np.array(cut_off)

    # Create a boolean mask for non-zero values in the value column (index 1)
    nonzero = fp[:, 1] != 0
    nonzero_indices = np.where(nonzero)[0]

    if nonzero_indices.size:
        # Split indices into contiguous groups
        groups = np.split(nonzero_indices, np.where(np.diff(nonzero_indices) != 1)[0] + 1)
        for group in groups:
            drift_len = group.size
            # If the contiguous window is longer than the maximum reference window,
            # set the value to 0 for every element in this window.
            if drift_len -refrence_window_maxlen > 336: #one week drift minimum
                fp[group[-refrence_window_maxlen:], 1] = 0

    # Update the filtered predictions list in-place with the modified [index, value] pairs
    cut_off[:] = fp.tolist()
    return cut_off



class OptunaOptimizedDetectorRunner:
    def __init__(self, base_model: callable, parameters: list, optuna_settings: dict = None, metric: callable = None, benchmark_metrics: bool = False, n_training_samples: int = None, cut_off: bool = False, clear_window: bool = False):
        self.base_model = base_model
        self.parameters = parameters
        defaults = {"n_trials": 50}
        if optuna_settings:
            defaults.update(optuna_settings)
        self.optuna_settings = defaults
        self.metric = metric
        self.benchmark_metrics = benchmark_metrics
        self.n_training_samples = n_training_samples
        self.cut_off = cut_off
        self.clear_window = clear_window
        self.best_config = None
        self.best_fitness = None
        self.logger = None

    def evaluate_individual(self, config: dict, stream, trial) -> float:
        """
        Evaluate a candidate configuration by running the detector on the stream.
        The stream is expected to yield tuples: (features, label)
        where label is taken from the last column ("class") and is 0 (no drift) or 1 (drift).
        
        For evaluation, we process the stream in windows of size equal to the model's step_size.
        For each window, the ground truth label is set to 1 if any sample in the window has label==1,
        otherwise 0. The model’s prediction for the window (obtained via its update(buffer) method)
        is compared against this aggregated ground truth.
        
        You can supply your own metric via the constructor. If none is provided, accuracy is used.
        """
        # Initialize model instance and buffers.
        model_instance = self.base_model(**config)
        predictions = []
        buffer = []
        groundt_truth = []

        if self.benchmark_metrics:
            classifiers_labels = []
            classifiers_drifts = []
            classifiers_predictions = []
            classifiers = Classifiers()
        

        # Process the rest of the stream in windows (of length = step_size).
        for i, (sample, label) in enumerate(stream):
            if self.benchmark_metrics & i != 0:
                classifiers_predictions.append(classifiers.predict(sample))
                classifiers_labels.append(label)
            groundt_truth.append(label)
            if model_instance.window_len == len(model_instance.data_window):
                buffer.append(np.fromiter(sample.values(), dtype=float))
                if len(buffer) >= model_instance.step_size:
                    drift_detected = model_instance.update(buffer, self.clear_window)
                    if self.benchmark_metrics & drift_detected:
                        classifiers.reset()
                        classifiers_drifts.append(i)
                    predictions.append([i,1] if drift_detected else [i,0])
                    buffer.clear()
            else:
                if len(model_instance.data_window) + model_instance.step_size == model_instance.window_len:
                    buffer.append(np.fromiter(sample.values(), dtype=float))
                    if len(buffer) == model_instance.step_size:
                        drift_detected = model_instance.update(buffer, self.clear_window)
                        if self.benchmark_metrics & drift_detected:
                            classifiers.reset()
                            classifiers_drifts.append(i)
                        predictions.append([i,1] if drift_detected else [i,0])
                        # Aggregate ground truth for this window:
                        buffer.clear()
                else:
                    model_instance.data_window.append(np.fromiter(sample.values(), dtype=float))
        
            if self.benchmark_metrics: 
                classifiers.fit(sample, label, nonadaptive=i < self.n_training_samples)
        print("======================  CURRERNT DETECTOR: ", model_instance.__class__.__name__, "======================")
        if self.benchmark_metrics:
            metrics = get_metrics(stream, classifiers_drifts, classifiers_labels, classifiers_predictions)
            print("*** BENCHMARK METRICS: ", metrics)
        fill_pred = fill_predictions(predictions, model_instance.step_size, model_instance.recent_window.maxlen)
        if self.cut_off:
            fill_pred = refrence_cutoff(fill_pred, model_instance.reference_window.maxlen)
        # Compute the evaluation metric.
        if self.metric is not None:
            if self.metric == "lpd_ht":
                score = metrics.lpd[0]
            elif self.metric == "acc_ht_dd":
                score = metrics.accuracies[2]
            else: # For f1 and precision
                score = self.metric(fill_pred, groundt_truth)
        else:
            # Default: accuracy calculation.
            correct = 0
            for pred, index in predictions:
                if (pred != 0 and groundt_truth[index] != 0) or (pred == 0 and groundt_truth[index] == 0):
                    correct += 1
                score = correct / len(predictions) if predictions else 0.0
        # Save the predictions list as a user attribute on this trial.
        trial.set_user_attr("accuracy",accuracy_metric(fill_pred, groundt_truth))
        trial.set_user_attr("f1", f1_metric(fill_pred, groundt_truth))
        trial.set_user_attr("predictions", predictions)
        trial.set_user_attr("Cut off", self.cut_off)
        trial.set_user_attr("Confussion matrix", confusion_matrix(fill_pred, groundt_truth))
        trial.set_user_attr("Clear window", self.clear_window)
        if self.benchmark_metrics:
            trial.set_user_attr("acc (ht-no dd)", metrics.accuracies[0])
            trial.set_user_attr("acc (nb-no dd)", metrics.accuracies[1])
            trial.set_user_attr("acc (ht-dd)", metrics.accuracies[2])
            trial.set_user_attr("acc (nb-dd)", metrics.accuracies[3])
            trial.set_user_attr("lpd (ht)", metrics.lpd[0])
            trial.set_user_attr("lpd (nb)", metrics.lpd[1])
            trial.set_user_attr("f1 (ht-no dd)", metrics.f1_scores[0])
            trial.set_user_attr("f1 (nb-no dd)", metrics.f1_scores[1])
            trial.set_user_attr("f1 (ht-dd)", metrics.f1_scores[2])
            trial.set_user_attr("f1 (nb-dd)", metrics.f1_scores[3])
            trial.set_user_attr("classifier_mtfa", metrics.mtfa)
            trial.set_user_attr("classifier_mtr", metrics.mtr)
            trial.set_user_attr("classifier_mtd", metrics.mtd)
            trial.set_user_attr("classifier_mdr", metrics.mdr)
        print("*** CONFUSION MATRIX: ",confusion_matrix(fill_pred, groundt_truth))
        #print("Accuracy: ", accuracy_metric(fill_pred, groundt_truth))

# acc [0] == accuracy of Hoeffding tree without drift detector,
# acc [1] == accuracy of naive Bayes without drift detector,
# acc [2] == accuracy of Hoeffding tree with drift detector,
# acc [3] == accuracy of naive Bayes with drift detector, 

        return score

    def optimize(self, stream) -> dict:
        def objective(config, trial):
            return self.evaluate_individual(config, stream, trial)

        optimizer = OptunaOptimizer(
            objective=objective,
            parameters=self.parameters,
            n_trials=self.optuna_settings.get("n_trials", 50),
            storage=self.optuna_settings.get("storage", "sqlite:///optuna.db"),
            study_name=self.optuna_settings.get("study_name", "optuna_study")
        )
        best_config, best_fitness = optimizer.optimize()
        self.best_config = best_config
        self.best_fitness = best_fitness
        return best_config

    def run(self, stream, experiment_name, n_training_samples=1000, verbose=False):
        """
        Optimize hyperparameters with Optuna and then run the model with the best configuration.
        """
        best_config = self.optimize(stream)
        print(f"_________________________finished optimization for model.__________________________")