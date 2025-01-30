import sys
import time

from config import Configuration
from ensemble import ensemble


def main():
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
    else:
        experiment_name = int(time.time())
  
    for stream in Configuration.streams:
        for model in Configuration.models:
            model.optimize(stream, experiment_name, Configuration.n_training_samples, verbose=True) 
            # TODO: goal: model.model_run(stream, experiment_name, step_size, verbose=True)
    ensemble(experiment_name)


if __name__ == "__main__":
    main()
    print("Done")

