import re
import json
import logging
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import List
from tqdm import tqdm
from pandarallel import pandarallel
from .instance_dataset import InstanceDataset


class DatasetFactory(Dataset):
    def __init__(self,
                 data_dir,
                 sr=16000,
                 mode='train',
                 rank=0,
                 preload_data=False,
                 transform=None,
                 nb_workers=4):
        self.df = None
        self.data_dir = data_dir
        self.mode = mode
        self.rank = rank
        self.sr = sr
        self.chars_to_ignore = r'[,?.!\-;:"“%\'�]'
        self.transform = transform
        self.preload_data = preload_data
        self.nb_workers = nb_workers
        # functions
        self.df = self.create_df()
        self.preprocess_df()

    @abstractmethod
    def create_df(self) -> pd.DataFrame:
        pass

    def preprocess_df(self, min_duration=-np.inf, max_duration=np.inf):
        pandarallel.initialize(progress_bar=False, nb_workers=self.nb_workers)

        logging.info('Excute Preprocessing')

        if min_duration == -np.inf or max_duration == np.inf:
            if self.rank == 0 and 'duration' not in self.df.columns:
                logging.info("Generate duration column")
                self.df['duration'] = (self.df['path']
                                       .parallel_apply(lambda filename: librosa.get_duration(path=filename)))

            mask = (self.df['duration'] <= max_duration) & (self.df['duration'] >= min_duration)
            self.df = self.df[mask]

        logging.info("Remove special characters")
        self.df['transcript'] = self.df['transcript'].parallel_apply(self._remove_special_characters)

        if self.preload_data:
            if self.rank == 0:
                print(f"\n*****Preloading {len(self.df)} data*****")
            self.df['wav'] = self.df['path'].parallel_apply(lambda filepath: load_wav(filepath, sr=self.sr))

    def _remove_special_characters(self, transcript) -> str:
        transcript = re.sub(self.chars_to_ignore, '', transcript).lower()
        return transcript

    def get_vocab_dict(self, special_tokens) -> dict[str, int]:
        all_text = " ".join(list(self.df["transcript"]))

        for v in special_tokens.values():
            all_text = all_text.replace(v, '')

        vocab_list = list(set(all_text))
        vocab_list.sort()
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]

        for v in special_tokens.values():
            vocab_dict[v] = len(vocab_dict)
        return vocab_dict

    def save_vocab_dict(self, special_tokens, save_path: str = "vocab.json"):
        vocab_dict = self.get_vocab_dict(special_tokens)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    def preload_dataset(self, paths, sr) -> List:
        wavs = []
        print("Preloading {} data".format(self.mode))
        for path in tqdm(paths, total=len(paths)):
            wav = load_wav(path, sr)
            wavs += [wav]
        return wavs

    def export_csv(self, filename="merged_dataset.csv"):
        self.df.to_csv(filename, index=False, encoding="utf-8")

    def get_dataset(self, split=False, test_size=0.1, val_size=0.1, random_state=42):
        """
        Returns:
            - A single `InstanceDataset` if `split=False`
            - A tuple (train_ds, val_ds, test_ds) if `split=True`
        """
        if not split:
            print(self.df.shape)
            return InstanceDataset(self.df, self.sr, self.preload_data, self.transform)

        train_val_df, test_df = train_test_split(self.df, test_size=test_size, random_state=random_state)
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(train_val_df, test_size=val_ratio, random_state=random_state)

        train_ds = InstanceDataset(train_df.reset_index(drop=True), self.sr, self.preload_data, self.transform)
        val_ds = InstanceDataset(val_df.reset_index(drop=True), self.sr, self.preload_data, self.transform)
        test_ds = InstanceDataset(test_df.reset_index(drop=True), self.sr, self.preload_data, self.transform)

        return train_ds, val_ds, test_ds