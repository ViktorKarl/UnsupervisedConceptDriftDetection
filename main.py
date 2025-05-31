import sys
import time
from config import Configuration
import logging_config       # ← config is applied here
from logging_config import logger

# Configure root logger once (usually at program entrypoint)

def main():
    experiment_name = sys.argv[1] if len(sys.argv) > 1 else int(time.time())
    logger.info("Run the models.")
    for stream in Configuration.streams:
        for model_runner in Configuration.models:
            
            logger.info(f"Running model: {model_runner.base_model.__name__}")

            #update the study name to include the model name
            model_runner.optuna_settings['study_name'] = Configuration.study_name + f"_{model_runner.base_model.__name__}" + f"_{stream.__class__.__name__}"
                
            # Optimize hyperparameters and run the model on the stream.
            try:
                model_runner.run(stream, experiment_name, Configuration, verbose=True)
            except Exception:
                logger.error(
                    "Model %s on dataset %s failed", 
                    model_runner.base_model.__name__, 
                    stream.__class__.__name__, 
                    exc_info=True
                )
            else:
                logger.info(f"Success runned: model: {model_runner.base_model.__name__}, dataset: {stream.__class__.__name__}")

    logger.info("Finished with the models")

if __name__ == "__main__":
    main()
    print("Done")
