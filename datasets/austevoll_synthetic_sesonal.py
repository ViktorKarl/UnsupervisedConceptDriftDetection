from os import path
from river.datasets import base
from river import stream

class AustevollSyntheticSesonal(base.FileDataset):
    def __init__(self, directory_path: str = "datasets/files"):
        super().__init__(
            n_samples=350687,
            n_features=4,
            task=base.MULTI_CLF,
            filename="synthetic_drift_20_years_sesonal.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        converters = {
            "Pressure": float,
            "Conductivity": float,
            "Temperature": float,
            "Salinity": float,
            "class": int,
        }
        return stream.iter_csv(
            self.full_path,
            target="class",
            converters=converters,
            drop=["Date"]  # Assuming the date column is named "date"
        )
