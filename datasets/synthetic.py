from os import path

from river.datasets import base
from river import stream


class Synthetic(base.FileDataset):
    def __init__(
        self,
        directory_path: str = "datasets/files",
    ):
        super().__init__(
            n_samples=17_521,
            n_features=5,
            task=base.MULTI_CLF,
            filename="SYNTHETIC.csv",
        )
        self.full_path = path.join(directory_path, self.filename)

    def __iter__(self):
        #converters = {f"attribute{i}": float for i in range(1, 5)}
        # temperature,humidity,wind_speed,precipitation,class
        converters = {'temperature': float, 'humidity': float, 'wind_speed': float, 'precipitation': float}
        converters["class"] = int
        return stream.iter_csv(
            self.full_path,
            target="class",
            converters=converters,
        )
