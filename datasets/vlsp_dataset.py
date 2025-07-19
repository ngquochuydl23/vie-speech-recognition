import os
import pandas as pd
from .dataset_factory import DatasetFactory


class VLSPDatasetFactory(DatasetFactory):
    def create_df(self):
        data = []
        for file in os.listdir(os.path.join(self.data_dir, 'vlsp2020_train_set_02')):
            if file.endswith('.wav'):
                base_name = os.path.splitext(file)[0]
                wav_path = os.path.join(os.path.join(self.data_dir, 'vlsp2020_train_set_02'), file)
                label_path = os.path.join(os.path.join(self.data_dir, 'vlsp2020_train_set_02'), base_name + '.txt')

                if os.path.isfile(label_path):
                    with open(label_path, 'r', encoding='utf-8') as f:
                        transcript = f.read().strip()
                    data.append([transcript, wav_path])
        return pd.DataFrame(data, columns=["transcript", "path"])
