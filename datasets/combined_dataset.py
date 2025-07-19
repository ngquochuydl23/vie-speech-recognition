from datasets.vivos_dataset import VIVOSDatasetFactory
from datasets.vlsp_dataset import VLSPDatasetFactory
import pandas as pd
from .dataset_factory import DatasetFactory


VLSP_DATASET_DIR = './vlsp-2020'
VLSP_DATA_DF = './vlsp-2020/dataset.csv'
VIVOS_DATASET_DIR = './vivos'
VIVOS_DF_PATH = './vivos/dataset.csv'

class CombinedDatasetFactory(DatasetFactory):
    def create_df(self):
        return pd.read_csv('./combined_dataset.csv')

if __name__ == '__main__':
    vivos_dataset_factory = VIVOSDatasetFactory(data_dir=VIVOS_DATASET_DIR)
    vlsp_dataset_factory = VLSPDatasetFactory(data_dir=VLSP_DATASET_DIR)

    vlsp_dataset_factory.export_csv(VLSP_DATA_DF)
    vivos_dataset_factory.export_csv(VIVOS_DF_PATH)

    vlsp_df = pd.read_csv(VLSP_DATA_DF)
    vivos_df = pd.read_csv(VIVOS_DF_PATH)
    combined_df = pd.concat([vlsp_df, vivos_df], ignore_index=True)
