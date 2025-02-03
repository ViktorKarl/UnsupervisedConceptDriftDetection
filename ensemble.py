import os
import pandas as pd
from config import Configuration

from typing import List, Optional



# NB! As of now, if there are muliple runs for each algorithm, the ensemble will only consider the last run.
# This means that there is only need for one run for each algorithm.
# And that the algorithms should already be optimized befor running the ensemble.

# This is the same threshold for the ensemble to say if a drift was found or not
threshold = 0.5

def ensemble(experiment_name):
    all_streams = {}
    for stream in Configuration.streams:
        # Determine a name for the stream.
        stream_name = getattr(stream, 'name', stream.__class__.__name__)
        found_drifts = {}

        for model in Configuration.models:
            # Get the model name from its base_model class
            model_name = model.base_model.__name__
            found_drifts[model_name] = []
            # Construct the file path
            csv_path = f"results/{stream_name}/{model_name}_{experiment_name}.csv"

            try:
                # Load the CSV file into a DataFrame
                df = pd.read_csv(csv_path)
                print(f"Loaded {csv_path}")

                # Get the drifts from the last row
                drifts = df.iloc[-1]['drifts']
                found_drifts[model_name].append(drifts)

            except FileNotFoundError:
                print(f"File not found: {csv_path}")
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
        all_streams[stream_name] = found_drifts

        number_of_detectors = len(Configuration.models)
        counter_drifts = 0
        for model_name, drifts_list in found_drifts.items():
            if len(drifts_list) > 0 and drifts_list[0] not in [None, "[]", ""]:
                counter_drifts += 1

        if (number_of_detectors > 0) and ((counter_drifts/number_of_detectors) >= threshold):
            print(f"Drifts found in {stream_name} by {counter_drifts} detectors out of {number_of_detectors}")
        else:
            print(f"No drifts found in {stream_name}")

    # Print all_streams for debug
    print(all_streams)

    # Now we want to save all_streams into a CSV in results/ensemble/
    # Ensure the directory exists
    output_dir = "results/ensemble"
    os.makedirs(output_dir, exist_ok=True)

    # Flatten all_streams into a DataFrame
    # Rows: stream_name, model_name, drifts
    rows = []
    for stream_name, models_dict in all_streams.items():
        for model_name, drifts_list in models_dict.items():
            # drifts_list is a list, usually with one element
            # Convert it to string or keep as-is
            # If empty, store something like '[]'
            drifts_str = drifts_list[0] if (len(drifts_list) > 0) else "[]"
            rows.append({
                'stream_name': stream_name,
                'model_name': model_name,
                'drifts': drifts_str
            })

    df_output = pd.DataFrame(rows)
    output_csv = f"{output_dir}/{experiment_name}.csv"
    df_output.to_csv(output_csv, index=False)
    print(f"Saved ensemble results to {output_csv}")







#########################################################################


class ensemble_class:
    def __init__(self,
                config: Configuration,
                experiment_name: str,
                thread: bool = False):
        self.config = config
        self.experiment_name = experiment_name
        
    def init_detector(self):
        pass

    def run_ensemble(self):
        # Load the results from the models
        # For each stream, for each model, get the drifts
        # If a drift is found by a majority of the models, then it is a drift
        # Save the results to a CSV file
        pass


