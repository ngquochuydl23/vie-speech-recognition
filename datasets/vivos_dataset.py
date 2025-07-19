import os
import pandas as pd
import logging
from .dataset_factory import DatasetFactory


class VIVOSDatasetFactory(DatasetFactory):
    def create_df(self):
        data = []
        logging.info('Loop train-data')
        with open(os.path.join(self.data_dir, 'train', 'prompts.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    wav_id = parts[0]
                    transcript = parts[1]
                    speaker_folder = wav_id.split('_')[0]
                    wav_path = os.path.join(self.data_dir, 'train', 'waves', speaker_folder, wav_id + '.wav')
                    data.append({
                        'transcript': transcript,
                        'path': wav_path
                    })
        with open(os.path.join(self.data_dir, 'test', 'prompts.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    wav_id = parts[0]
                    transcript = parts[1]
                    speaker_folder = wav_id.split('_')[0]
                    wav_path = os.path.join(self.data_dir, 'test' , 'waves', speaker_folder, wav_id + '.wav')
                    data.append({
                        'transcript': transcript,
                        'path': wav_path
                    })
        return pd.DataFrame(data, columns=["transcript", "path"])

