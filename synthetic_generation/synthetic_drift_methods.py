import pandas as pd
import numpy as np

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
    
    # For each numeric column, choose a random offset
    for col in columns:
        offset = rng.normal(loc=0, scale=0.3)  # normal distribution
        offset *= significance
        df.iloc[position:, df.columns.get_loc(col)] += offset

def apply_gradual_drift(df, start_pos, end_pos, significance=1.0, columns=None, random_state=None):
    """
    Apply a gradual drift from `start_pos` to `end_pos` (linear ramp).
    The drift grows from offset=0 at start_pos to offset=some_value at end_pos.
    Also sets `drift` = 1 for rows [start_pos, end_pos).
    """
    if random_state is None:
        rng = np.random.RandomState()
    elif isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    else:
        rng = random_state

    if end_pos <= start_pos:
        return  # no drift needed
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.difference(["drift"])

    length = end_pos - start_pos
    
    # Mark these rows in the 'drift' label
    df.iloc[start_pos:end_pos, df.columns.get_loc("drift")] = 1

    for col in columns:
        offset = rng.normal(loc=0, scale=0.5)  # normal distribution
        offset *= significance

        # linearly ramp from 0% offset at start_pos to 100% offset at end_pos
        for i in range(start_pos, end_pos):
            progress = (i - start_pos) / float(length)
            df.iloc[i, df.columns.get_loc(col)] += offset * progress



def add_synthetic_drifts(
    df,
    number_of_gradual_drifts=1,
    number_of_abrupt_drifts=1,
    significance=1.0,
    random_state=None,
    columns=None):
    
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


    # For now the drift happends at random places in the dataset. The problem with that is the drifts can overlap and stack on top of each other.
    # This is not realistic. We need to make sure that the drifts are not overlapping.
    # We can do this by creating a list of all the possible positions where a drift can happen and then randomly select from this list.
    # When a position is selected we remove the position and the surrounding positions from the list.
    

    # 2) Insert abrupt drifts
    for i in range(number_of_abrupt_drifts):
        pos = rng.randint(low=0, high=n_rows)
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

        # Define the minimum and maximum drift width as 10% and 20% of total length
        min_width = int(0.05 * n_rows)   # 10% minimum
        max_width = int(0.2 * n_rows)   # 20% maximum

        start = rng.randint(low=0, high=n_rows - max_width)
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
            random_state=rng
        )
