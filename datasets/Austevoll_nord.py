from os import path

from river.datasets import base
from river import stream

class AustevollNord(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=6986,
            n_features=4,
            task=base.MULTI_CLF,
            filename="Austevoll_Nord_astrid_filtered_crop.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        return stream.iter_csv(
            self.full_path,
            target="Conductivity",
            drop=["Date"],  # Exclude the Date column from features
            parse_dates={},  # No date parsing needed as Date column is dropped
            converters={
                "Temperature": float,
                "Salinity": float,
                "Pressure": float,
                "Conductivity": float
            }
            )
        
    



