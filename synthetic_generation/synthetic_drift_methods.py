import pandas as pd
import numpy as np
import random

# Function to sample a value from a normal distribution excluding a middle range for the drift to be significant
def sample_excluding_middle(rng, loc=0.3, scale=0.5, lower=-0.5, upper=0.5):
    while True:
        val = rng.normal(loc, scale)
        if not (lower < val < upper):
            return val


def load_time_series(csv_path, time_col=None, parse_dates=True):
    """
    Load a multivariate time-series from a CSV file into a pandas DataFrame.
    
    :param csv_path: Path to the CSV file
    :param time_col: (Optional) name of a timestamp column to set as index
    :param parse_dates: Whether to parse the time_col as datetime
    :return: A pandas DataFrame with a DateTime index (if time_col is provided),
             or a standard integer index otherwise.
    """
    if time_col:
        df = pd.read_csv(csv_path, parse_dates=[time_col])
        df.set_index(time_col, inplace=True)
    else:
        df = pd.read_csv(csv_path)
    return df

def apply_abrupt_drift(df, position, significance=1.0, columns=None, random_state=None):
    """
    Apply an abrupt drift at `position` by adding a random offset (scaled by `significance`)
    to the data from `position` onward, in-place.
    Also sets `drift` = 1 for those rows [position, end).
    """
    if random_state is None:
        rng = np.random.RandomState()
    elif isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    else:
        rng = random_state
    
    if columns is None:
        # Use all numeric columns except 'drift' if it exists
        columns = df.select_dtypes(include=[np.number]).columns.difference(["drift"])
    
    # Mark these rows in the 'drift' label
    df.iloc[position, df.columns.get_loc("drift")] = 1
    
    # Apply drift to a random column
    random_col = rng.choice(columns)
    # Generate a random offset for the drift
    # Draws a random sample from a normal (Gaussian) distribution with:
    # loc = 0 as the mean (the center of the distribution), and
    # scale = 0.3 as the standard deviation (how spread out the distribution is).
    offset = sample_excluding_middle(rng, loc=0, scale=0.3) * significance
    #offset = rng.normal(loc=0, scale=0.3)
    # Scale the offset by the significance
    offset *= significance
    # Apply the drift to the data
    df.iloc[position:, df.columns.get_loc(random_col)] += offset

def apply_gradual_drift(df, start_pos, end_pos, significance=1.0, columns=None, random_state=None, sigmoid=False):
    if random_state is None:
        rng = np.random.RandomState()
    elif isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    else:
        rng = random_state

    if end_pos <= start_pos:
        return
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.difference(["drift"])

    length = end_pos - start_pos
    df.iloc[start_pos:end_pos, df.columns.get_loc("drift")] = 1

    # Apply drift to a random column
    random_col = rng.choice(columns)
    # Generate a random offset for the drift
    # Draws a random sample from a normal (Gaussian) distribution with:
    # loc = 0 as the mean (the center of the distribution), and
    # scale = 0.5 as the standard deviation (how spread out the distribution is).
    offset = sample_excluding_middle(rng, loc=0, scale=0.5) * significance
    #offset = rng.normal(loc=0, scale=0.5) * significance
    # Intervallene for sannsynligheten til -0,7 til 1,3 og -1,2 til 1,8 for henholdsvis 95% og 99,7% sannsynlighet

    # Apply the drift to the data
    for i in range(start_pos, end_pos):
        progress = (i - start_pos) / float(length)
        if sigmoid:
            drift_value = offset / (1 + np.exp(-12*(progress - 0.5)))
        else:
            drift_value = offset * progress
        df.iloc[i, df.columns.get_loc(random_col)] += drift_value

    df.iloc[end_pos:, df.columns.get_loc(random_col)] += offset



def add_synthetic_drifts(
    df,
    number_of_gradual_drifts=1,
    number_of_abrupt_drifts=1,
    significance=1.0,
    random_state=None,
    columns=None,
    sigmoid=False):

    
    """
    Add a specified number of gradual and abrupt drifts to the DataFrame in-place,
    printing what drift type is applied and where, and labeling drift-affected rows
    by setting 'drift' = 1 in those time intervals.
    """
    if random_state is None:
        rng = np.random.RandomState()
    elif isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    else:
        rng = random_state
    
    n_rows = len(df)
    
    # 1) Initialize a 'drift' label column if it doesn't exist
    #    This will be 0 for all rows initially
    if "drift" not in df.columns:
        df["drift"] = 0

    drift_window_size = n_rows // (number_of_abrupt_drifts + number_of_gradual_drifts)

    subspaces = list()
    for i in range(1, (number_of_abrupt_drifts+number_of_gradual_drifts)+1):
        start = (i * drift_window_size) - drift_window_size
        end = (i * drift_window_size)
        if start != 0:
            start += 1
        subspaces.append([start, end])

    print(subspaces)
    random.shuffle(subspaces)
    print(subspaces)

    # For now the drift happends at random places in the dataset. The problem with that is the drifts can overlap and stack on top of each other.
    # This is not realistic. We need to make sure that the drifts are not overlapping.
    # We can do this by creating a list of all the possible positions where a drift can happen and then randomly select from this list.
    # When a position is selected we remove the position and the surrounding positions from the list.
    

    # 2) Insert abrupt drifts
    for i in range(number_of_abrupt_drifts):
        current_drift_window = subspaces.pop()
        pos = rng.randint(low=current_drift_window[0], high=current_drift_window[1])
        print(f"Applying abrupt drift {i+1}/{number_of_abrupt_drifts} at position={pos}")
        apply_abrupt_drift(
            df, 
            position=pos, 
            significance=significance, 
            columns=columns,
            random_state=rng
        )
    
    # 3)  Insert gradual drifts
    for j in range(number_of_gradual_drifts):
        current_drift_window = subspaces.pop()
        # Define the minimum and maximum drift width as 10% and 20% of total length
        min_width = int(0.1 * drift_window_size)   # 10% minimum
        max_width = int(0.8 * drift_window_size)   # 80% maximum

        start = rng.randint(low=current_drift_window[0], high=current_drift_window[1] - max_width)
        # In case the dataset is small, ensure at least 1 row
        if min_width < 1:
            min_width = 1

        if max_width < 1:
            max_width = 1

        # Randomly pick a width in [min_width, max_width]
        if min_width < max_width:
            width = rng.randint(low=min_width, high=max_width)
        else:
            width = min_width  # fallback if n_rows is too small

        end = min(start + width, n_rows)
        
        print(f"Applying gradual drift {j+1}/{number_of_gradual_drifts} from {start} to {end}")
        apply_gradual_drift(
            df,
            start_pos=start,
            end_pos=end,
            significance=significance,
            columns=columns,
            random_state=rng,
            sigmoid=sigmoid
        )
