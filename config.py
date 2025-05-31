from datasets import (
    Electricity,
    InsectsAbruptBalanced,
    InsectsGradualBalanced,
    InsectsIncrementalAbruptBalanced,
    InsectsIncrementalBalanced,
    InsectsIncrementalReoccurringBalanced,
    NOAAWeather,
    OutdoorObjects,
    Synthetic,
    PokerHand,
    Powersupply,
    RialtoBridgeTimelapse,
    SineClusters,
    WaveformDrift2,
    AustevollNord,
    AustevollSyntheticSesonal,
    AustevollSyntheticNonSesonal
)
from detectors import *
from optimize_optuna.model_run import OptunaOptimizedDetectorRunner
from run_detectors.parameter import Parameter
from optimize_optuna.evaluation_metrics import accuracy_metric, f1_metric, precision_metric

class Configuration:
    study_name = "F1_score" # <-- Set your desired study name here.
    optuna_settings = {
         'n_trials': 20, # Number of trials to run
         'study_name': '',  
         'storage': 'sqlite:///<path_to_lite_optuna.db>',  # Path to the SQLite database file

    }

    # optuna-dashboard 

    # Choose the evaluation metric function.
    #evaluation_metric = accuracy_metric  
    evaluation_metric = f1_metric
    #evaluation_metric = precision_metric
    #evaluation_metric = "lpd_ht"
    #evaluation_metric = "acc_ht_dd"

    cut_off = True # Flyttet fra model baseclass
    clear_window = False # Flyttet fra model baseclass


    benchmark_metrics = True  #lpd hoffening tree
    n_training_samples = 1000  # hoffening tree training samples
    

 

    # Define which datasets to use.
    stream_selection = {
        "Electricity": False,
        "InsectsAbruptBalanced": False,
        "InsectsGradualBalanced": False,
        "InsectsIncrementalAbruptBalanced": False,
        "InsectsIncrementalBalanced": False,
        "InsectsIncrementalReoccurringBalanced": False,
        "NOAAWeather": False,
        "OutdoorObjects": False,
        "PokerHand": False,
        "Powersupply": False,
        "RialtoBridgeTimelapse": False,
        "SineClusters": False,
        "WaveformDrift2": False,
        "Synthetic": False,
        "AustevollNord": False,
        "AustevollSyntheticSesonal": False,
        "AustevollSyntheticNonSesonal": True
    }

    # Define which models (detectors) to optimize.
    model_selection = {
        "BayesianNonparametricDetectionMethod": True,    # refactored
        "ClusteredStatisticalTestDriftDetectionMethod": False,
        "DiscriminativeDriftDetector2019": True,         # refactored
        "ImageBasedDriftDetector": False,
        "OneClassDriftDetector": False,
        "SemiParametricLogLikelihood": True,             # refactored
        "UDetect_Disjoint": False,
        "UDetect_NonDisjoint": False,
        "KullbackLeiblerDistanceDetector": True,         # refactored
        "JensenShannonDistanceDetector": True,           # refactored
        "HellingerDistanceDetector": True                # refactored
    }

    # Create the list of streams based on the stream selection.
    streams = []
    if stream_selection["Electricity"]:
        streams.append(Electricity())
    if stream_selection["InsectsAbruptBalanced"]:
        streams.append(InsectsAbruptBalanced())
    if stream_selection["InsectsGradualBalanced"]:
        streams.append(InsectsGradualBalanced())
    if stream_selection["InsectsIncrementalAbruptBalanced"]:
        streams.append(InsectsIncrementalAbruptBalanced())
    if stream_selection["InsectsIncrementalBalanced"]:
        streams.append(InsectsIncrementalBalanced())
    if stream_selection["InsectsIncrementalReoccurringBalanced"]:
        streams.append(InsectsIncrementalReoccurringBalanced())
    if stream_selection["NOAAWeather"]:
        streams.append(NOAAWeather())
    if stream_selection["OutdoorObjects"]:
        streams.append(OutdoorObjects())
    if stream_selection["PokerHand"]:
        streams.append(PokerHand())
    if stream_selection["Powersupply"]:
        streams.append(Powersupply())
    if stream_selection["RialtoBridgeTimelapse"]:
        streams.append(RialtoBridgeTimelapse())
    if stream_selection["SineClusters"]:
        streams.append(SineClusters(drift_frequency=5000, stream_length=154987, seed=531874))
    if stream_selection["WaveformDrift2"]:
        streams.append(WaveformDrift2(drift_frequency=5000, stream_length=154987, seed=2401137))
    if stream_selection["Synthetic"]:
        streams.append(Synthetic())
    if stream_selection["AustevollNord"]:
        streams.append(AustevollNord())
    if stream_selection["AustevollSyntheticSesonal"]:
        streams.append(AustevollSyntheticSesonal())
    if stream_selection["AustevollSyntheticNonSesonal"]:
        streams.append(AustevollSyntheticNonSesonal())


    # Settings for the Optuna optimizer.
    week = 336
    day = 48
    hour = 2

     # Build the list of model runners using the chosen detectors and their hyperparameter search space.
    models = []
    if model_selection["BayesianNonparametricDetectionMethod"]:
        models.append(OptunaOptimizedDetectorRunner(
            base_model=BayesianNonparametricDetectionMethod,
            parameters=[
                Parameter("window_len", values=[week*3, week*15]),
                Parameter("step_size", values=[day, week*2]),
                Parameter("const", values=[0.1, 5.0]),
                Parameter("max_depth", values=[1, 12]),
                Parameter("threshold", values=[0.1, 2.0]),
            ],
            optuna_settings=optuna_settings,
            metric=evaluation_metric,
            benchmark_metrics=benchmark_metrics,
            n_training_samples = n_training_samples,
            cut_off = cut_off
        ))
    if model_selection["ClusteredStatisticalTestDriftDetectionMethod"]:
        models.append(OptunaOptimizedDetectorRunner(
            base_model=ClusteredStatisticalTestDriftDetectionMethod,
            parameters=[
                Parameter("n_samples", values=[250, 750]),
                Parameter("confidence", values=[0.01, 0.2]),
                Parameter("feature_proportion", values=[0.05, 0.2]),
                Parameter("n_clusters", values=[2, 3]),
            ],
            optuna_settings=optuna_settings,
            metric=evaluation_metric,
            benchmark_metrics=benchmark_metrics,
            n_training_samples = n_training_samples,
            cut_off = cut_off
        ))
    if model_selection["DiscriminativeDriftDetector2019"]:
        models.append(OptunaOptimizedDetectorRunner(
            base_model=DiscriminativeDriftDetector2019,
            parameters=[
                Parameter("n_reference_samples", values=[week*3, week*7]),
                Parameter("n_recent_samples", values=[week*2, week*7]),
                Parameter("step_size", values=[day, week*2]),
                Parameter("threshold", values=[0.15,3.0]),
            ],
            optuna_settings=optuna_settings,
            metric=evaluation_metric,
            benchmark_metrics=benchmark_metrics,
            n_training_samples = n_training_samples,
            cut_off = cut_off
        ))
    if model_selection["ImageBasedDriftDetector"]:
        models.append(OptunaOptimizedDetectorRunner(
            base_model=ImageBasedDriftDetector,
            parameters=[
                Parameter("n_samples", values=[100, 1000]),
                Parameter("n_permutations", values=[10, 40]),
                Parameter("update_interval", values=[50, 250]),
                Parameter("n_consecutive_deviations", values=[1, 4]),
            ],
            optuna_settings=optuna_settings,
            metric=evaluation_metric,
            benchmark_metrics=benchmark_metrics,
            n_training_samples = n_training_samples,
            cut_off = cut_off
        ))
    if model_selection["OneClassDriftDetector"]:
        models.append(OptunaOptimizedDetectorRunner(
            base_model=OneClassDriftDetector,
            parameters=[
                Parameter("n_samples", values=[100, 1000]),
                Parameter("threshold", values=[0.2, 0.5]),
                Parameter("outlier_detector_kwargs", values=[{"nu": 0.5, "kernel": "rbf", "gamma": "auto"}]),
            ],
            optuna_settings=optuna_settings,
            metric=evaluation_metric,
            benchmark_metrics=benchmark_metrics,
            n_training_samples = n_training_samples,
            cut_off = cut_off
        ))
    if model_selection["UDetect_Disjoint"]:
        models.append(OptunaOptimizedDetectorRunner(
            base_model=UDetect,
            parameters=[
                Parameter("n_windows", values=[25, 100]),
                Parameter("n_samples", values=[50, 500]),
                Parameter("disjoint_training_windows", values=[True]),
            ],
            optuna_settings=optuna_settings,
            metric=evaluation_metric,
            benchmark_metrics=benchmark_metrics,
            n_training_samples = n_training_samples,
            cut_off = cut_off
        ))
    if model_selection["UDetect_NonDisjoint"]:
        models.append(OptunaOptimizedDetectorRunner(
            base_model=UDetect,
            parameters=[
                Parameter("n_windows", values=[50, 250]),
                Parameter("n_samples", values=[100, 1000]),
                Parameter("disjoint_training_windows", values=[False]),
            ],
            optuna_settings=optuna_settings,
            metric=evaluation_metric,
            benchmark_metrics=benchmark_metrics,
            n_training_samples = n_training_samples,
            cut_off = cut_off
        ))
    if model_selection["KullbackLeiblerDistanceDetector"]:
        models.append(OptunaOptimizedDetectorRunner(
            base_model=KullbackLeiblerDistanceDetector,
            parameters=[
                Parameter("window_len", values=[week*3, week*15]),
                Parameter("threshold", values=[0.001, 3]),
                Parameter("step_size", values=[day, week*2]),
            ],
            optuna_settings=optuna_settings,
            metric=evaluation_metric,
            benchmark_metrics=benchmark_metrics,
            n_training_samples = n_training_samples,
            cut_off = cut_off
        ))
    if model_selection["JensenShannonDistanceDetector"]:
        models.append(OptunaOptimizedDetectorRunner(
            base_model=JensenShannonDistanceDetector,
            parameters=[
                Parameter("window_len", values=[week*3, week*15]),
                Parameter("threshold", values=[0.005, 3]),
                Parameter("step_size", values=[day, week*2]),
            ],
            optuna_settings=optuna_settings,
            metric=evaluation_metric,
            benchmark_metrics=benchmark_metrics,
            n_training_samples = n_training_samples,
            cut_off = cut_off
        ))
    if model_selection["HellingerDistanceDetector"]:
        models.append(OptunaOptimizedDetectorRunner(
            base_model=HellingerDistanceDetector,
            parameters=[
                Parameter("window_len", values=[week*3, week*15]),
                Parameter("threshold", values=[0.005, 3]),
                Parameter("step_size", values=[day, week*2]),
            ],
            optuna_settings=optuna_settings,
            metric=evaluation_metric,
            benchmark_metrics=benchmark_metrics,
            n_training_samples = n_training_samples,
            cut_off = cut_off
        ))
    if model_selection["SemiParametricLogLikelihood"]:
        models.append(OptunaOptimizedDetectorRunner(
            base_model=SemiParametricLogLikelihood,
            parameters=[
                Parameter("window_len", values=[week*3, week*15]),
                Parameter("step_size", values=[day, week*2]),
                Parameter("n_clusters", values=[2, 6]),
                Parameter("threshold", values=[0.005, 0.8])
            ],
            optuna_settings=optuna_settings,
            metric=evaluation_metric,
            benchmark_metrics=benchmark_metrics,
            n_training_samples = n_training_samples,
            cut_off = cut_off
        ))
