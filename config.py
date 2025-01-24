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
    AustevollNord
)
from detectors import *
from optimization.model_optimizer import ModelOptimizer
from optimization.parameter import Parameter


class Configuration:
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
        "Synthetic":False,
        "AustevollNord": True
    }

    model_selection = {
        "BayesianNonparametricDetectionMethod": True,
        "ClusteredStatisticalTestDriftDetectionMethod": False,
        "DiscriminativeDriftDetector2019": False,
        "ImageBasedDriftDetector": False,
        "OneClassDriftDetector": False,
        "SemiParametricLogLikelihood": False,
        "UDetect_Disjoint": False,
        "UDetect_NonDisjoint": False,
        "KullbackLeiblerDistanceDetector": False,
        "JensenShannonDistanceDetector": False,
        "HellingerDistanceDetector": False
    }

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
        streams.append(SineClusters(drift_frequency=5000, stream_length=154_987, seed=531874))
    if stream_selection["WaveformDrift2"]:
        streams.append(WaveformDrift2(drift_frequency=5000, stream_length=154_987, seed=2401137))
    if stream_selection["AustevollNord"]:
        streams.append(AustevollNord())

    # Our own insearted datasets: 
    if stream_selection["Synthetic"]:
        streams.append(Synthetic())


    n_training_samples = 1000

    models = []

    if model_selection["BayesianNonparametricDetectionMethod"]:
        models.append(ModelOptimizer(
            base_model=BayesianNonparametricDetectionMethod,
            parameters=[
                Parameter("n_samples", values=[500]),#, 1000]),
                Parameter("const", values=[0.5]),#, 1.0]),
                Parameter("max_depth", values=[2]),
                Parameter("threshold", values=[0.5]),
            ],
            seeds=None,
            n_runs=1,
        ))

    if model_selection["ClusteredStatisticalTestDriftDetectionMethod"]:
        models.append(ModelOptimizer(
            base_model=ClusteredStatisticalTestDriftDetectionMethod,
            parameters=[
                Parameter("n_samples", values=[500]),#, 1000]),
                Parameter("confidence", values=[0.1]),#, 0.01]),
                Parameter("feature_proportion", values=[0.1]),
                Parameter("n_clusters", values=[2]),
            ],
            seeds=None,
            n_runs=5,
        ))

    if model_selection["DiscriminativeDriftDetector2019"]:
        models.append(ModelOptimizer(
            base_model=DiscriminativeDriftDetector2019,
            parameters=[
                Parameter("n_reference_samples", values=[50, 125, 250, 500]),
                Parameter("recent_samples_proportion", values=[0.1, 0.5, 1.0]),
                Parameter("threshold", values=[0.6, 0.7, 0.8]),
            ],
            seeds=None,
            n_runs=5,
        ))

    if model_selection["ImageBasedDriftDetector"]:
        models.append(ModelOptimizer(
            base_model=ImageBasedDriftDetector,
            parameters=[
                Parameter("n_samples", values=[100, 250, 500, 1000]),
                Parameter("n_permutations", values=[10, 20, 40]),
                Parameter("update_interval", values=[50, 100, 250]),
                Parameter("n_consecutive_deviations", values=[1, 4]),
            ],
            seeds=None,
            n_runs=5,
        ))

    if model_selection["OneClassDriftDetector"]:
        models.append(ModelOptimizer(
            base_model=OneClassDriftDetector,
            parameters=[
                Parameter("n_samples", values=[100, 250, 500, 1000]),
                Parameter("threshold", values=[0.2, 0.3, 0.4, 0.5]),
                Parameter("outlier_detector_kwargs", value={"nu": 0.5, "kernel": "rbf", "gamma": "auto"})
            ],
            seeds=None,
            n_runs=1,
        ))

    if model_selection["SemiParametricLogLikelihood"]:
        models.append(ModelOptimizer(
            base_model=SemiParametricLogLikelihood,
            parameters=[
                Parameter("n_samples", values=[100, 250, 500, 1000]),
                Parameter("n_clusters", values=[2, 3]),
                Parameter("threshold", values=[0.05, 0.005]),
            ],
            seeds=None,
            n_runs=1,
        ))

    if model_selection["UDetect_Disjoint"]:
        models.append(ModelOptimizer(
            base_model=UDetect,
            parameters=[
                Parameter("n_windows", values=[25, 50, 100]),
                Parameter("n_samples", values=[50, 100, 250, 500]),
                Parameter("disjoint_training_windows", value=True)
            ],
            seeds=None,
            n_runs=1,
        ))

    if model_selection["UDetect_NonDisjoint"]:
        models.append(ModelOptimizer(
            base_model=UDetect,
            parameters=[
                Parameter("n_windows", values=[50, 100, 250]),
                Parameter("n_samples", values=[100, 250, 500, 1000]),
                Parameter("disjoint_training_windows", value=False)
            ],
            seeds=None,
            n_runs=1,
        ))
    if model_selection["KullbackLeiblerDistanceDetector"]:
        models.append(ModelOptimizer(
            base_model=KullbackLeiblerDistanceDetector,
            parameters=[
                Parameter("n_samples", values=[672]),
                Parameter("threshold", values=[0.05]),  #[0.01, 0.05, 0.1]),
            ],
            seeds=None,
            n_runs=1,
        ))
    if model_selection["JensenShannonDistanceDetector"]:
        models.append(ModelOptimizer(
            base_model=JensenShannonDistanceDetector,
            parameters=[
                Parameter("n_samples", values=[672, 1344]),
                Parameter("threshold", values=[0.1, 0.2, 0.3]),
            ],
            seeds=None,
            n_runs=1,
        ))
    if model_selection["HellingerDistanceDetector"]:
        models.append(ModelOptimizer(
            base_model=HellingerDistanceDetector,
            parameters=[
                Parameter("n_samples", values=[672]),
                Parameter("threshold", values=[0.1]),
            ],
            seeds=None,
            n_runs=1,
        ))
