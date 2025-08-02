import pandas as pd
from .dataset_factory import DatasetFactory

class CombinedDatasetFactory(DatasetFactory):
    def create_df(self):
        return pd.read_csv('./filtered_dataset.csv')

