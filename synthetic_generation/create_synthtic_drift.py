from synthetic_drift_methods import add_synthetic_drifts,load_time_series
import matplotlib.pyplot as plt
import numpy as np
import random
    
CSV_PATH = "synthetic_generation/data/SYNTHETIC.csv"
OUTPUT_PATH = "synthetic_generation/data/output/added_synthtic_drift.csv"
    
TIME_COL = None     # or "timestamp" if your CSV has a time column
SEED = random.randint(1, 1000)          # for reproducible random drifts
NUM_GRADUAL = 3     # number of gradual drifts
NUM_ABRUPT = 3      # number of abrupt drifts
SIGNIFICANCE = 2  # how large the drift offsets can be (the "severity")
SIGMOID = True     # use sigmoid function for gradual drifts, if false linear is used


# Load the original time series data
df_original = load_time_series(
    csv_path=CSV_PATH, 
    time_col=TIME_COL,
    parse_dates=True
)
# Drop the last column (if it's a label)
df_original = df_original.iloc[:, :-1]

print("Original DataFrame shape:", df_original.shape)
print(df_original.head())
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
)
    # Save the drifted data to a new CSV
df_drifted.to_csv(OUTPUT_PATH, index=bool(TIME_COL))
print(f"Saved drifted time series to: {OUTPUT_PATH}")

# Plot "Before Drift" vs. "After Drift" for one column 
# Pick the first numeric column or specify a particular column name
numeric_cols = df_original.select_dtypes(include=[np.number]).columns
if len(numeric_cols) == 0:
    print("No numeric columns found. Skipping plot.")
else:
    for column_to_plot in numeric_cols:
        plt.figure(figsize=(10, 6))
        plt.plot(df_original.index, df_original[column_to_plot], label="Before Drift")
        plt.plot(df_drifted.index, df_drifted[column_to_plot], label="After Drift")
        plt.plot(
            df_original.index,
            df_drifted[column_to_plot] - df_original[column_to_plot] + df_original[column_to_plot].mean(),
            label="Difference"
        )
        plt.title(f"Column '{column_to_plot}': Before vs. After Synthetic Drifts")
        plt.xlabel("Time / Index")
        plt.ylabel(column_to_plot)
        plt.legend()
        plt.tight_layout()
        plt.show()
