from synthetic_drift_methods import add_synthetic_drifts,load_time_series
import matplotlib.pyplot as plt
import numpy as np
import random

    
CSV_PATH = "synthetic_generation/data/calculated_conductivity_seasonal_completed.csv"
OUTPUT_PATH = "synthetic_generation/data/output/synthetic_drift_20_years_sesonal_3_sesonal_variates.csv"
    
TIME_COL = None     # or "timestamp" if your CSV has a time column
SEED = 607 #random.randint(1, 1000)          # for reproducible random drifts
NUM_GRADUAL = 10     # number of gradual drifts
NUM_ABRUPT = 0      # number of abrupt drifts
SIGNIFICANCE = 0.09  # how large the drift offsets can be (the "severity")
SIGMOID = True     # use sigmoid function for gradual drifts, if false linear is used
GRADUAL_SIZE_MIN = 48 * 7 * 2 # 2 Uker  # size of gradual drifts (in time steps) Here 480 is 1 day.
GRADUAL_SIZE_MAX = 48 * 7 * 4 * 2 # 2 Måneder  # size of gradual drifts (in time steps) Here 480 is 1 day.
CUT_EXTREMES = True  # cut the extremes of the drifts
CUT_RANGE = 0.1 # cut the extremes of the drifts by this fraction 0.1 = 10% of the drifts will be cut


# Load the original time series data
df_original = load_time_series(
    csv_path=CSV_PATH, 
    time_col=TIME_COL,
    parse_dates=True
)
# Drop the last column (if it's a label)
#df_original = df_original.iloc[:, :-1]

print("Original DataFrame shape:", df_original.shape)
print(df_original.head())
print("Seed for reproducible drifts:", SEED)
# Make a copy so we can compare "before" vs. "after" drift
df_drifted = df_original.copy()
add_synthetic_drifts(
    df=df_drifted,
    number_of_gradual_drifts=NUM_GRADUAL,
    number_of_abrupt_drifts=NUM_ABRUPT,
    significance=SIGNIFICANCE,
    random_state=SEED,
    columns=None,  # or specify specific columns
    sigmoid=SIGMOID, # for gradual drifts
    gradual_size_max = GRADUAL_SIZE_MAX,
    gradual_size_min = GRADUAL_SIZE_MIN,
    cut_extremes=CUT_EXTREMES,
    cut_range=CUT_RANGE
)
    # Save the drifted data to a new CSV
df_drifted.to_csv(OUTPUT_PATH, index=bool(TIME_COL))
print(f"Saved drifted time series to: {OUTPUT_PATH}")


# Ensure df_original and df_drifted have the same length + index alignment
N = len(df_original.index)

# Create a linear "year" axis, from 1 to 20
x = np.linspace(1, 20, N)

numeric_cols = df_original.select_dtypes(include=[np.number]).columns

if len(numeric_cols) == 0:
    print("No numeric columns found. Skipping plot.")
else:
    for column_to_plot in numeric_cols:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot "before" and "after" drift
        ax.plot(
            x, 
            df_drifted[column_to_plot], 
            label="With Drift"
        )
        ax.plot(
            x, 
            df_original[column_to_plot], 
            label="Before Drift"
        )
        
        # Plot the difference (shifted by the mean for easier visual comparison)
        difference = df_drifted[column_to_plot] - df_original[column_to_plot]
        ax.plot(
            x,
            difference + df_original[column_to_plot].mean(),
            color="#CC6183",  # or any color you wish
            label="Difference"
        )
        
        # Identify the start/end points of each drift period
        drift_series = df_drifted["class"].fillna(0)  # ensure no NaNs
        # A "drift start" is where we go from 0 -> 1
        drift_starts = drift_series[
            (drift_series.shift(1, fill_value=0) == 0) & (drift_series == 1)
        ].index
        # A "drift end" is where we go from 1 -> 0
        drift_ends = drift_series[
            (drift_series.shift(1, fill_value=0) == 1) & (drift_series == 0)
        ].index
        
        # Convert drift start/end indices to integer locations
        drift_starts_locs = [df_drifted.index.get_loc(s) for s in drift_starts]
        drift_ends_locs = [df_drifted.index.get_loc(e) for e in drift_ends]
        
        # Add dotted vertical lines for each drift start/end
        # We only label the first line of each type so the legend is cleaner
        first_start = True
        for loc in drift_starts_locs:
            label = "Drift Start" if first_start else None
            ax.axvline(x[loc], linestyle="--", color="green", alpha=0.9, label=label)
            first_start = False
        
        first_end = True
        for loc in drift_ends_locs:
            label = "Drift End" if first_end else None
            ax.axvline(x[loc], linestyle="--", color="red", alpha=0.9, label=label)
            first_end = False

        ax.set_xticks(range(1, 21))
    
        # Make a nice title/labels/legend
        ax.set_title(f"Column '{column_to_plot}': Before vs. After Synthetic Drifts")
        ax.set_xlabel("Year")
        if column_to_plot == "Temperature":
            ax.set_ylabel("Temperature [°C]")
        else:
            ax.set_ylabel(column_to_plot)
        ax.legend()
        
        plt.tight_layout()
        plt.show()