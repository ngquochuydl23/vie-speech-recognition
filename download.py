import os
import tarfile
import requests
from tqdm import tqdm
from logger import logging
from utils.tqdm_config import TQDMConfigs
from utils.dataset_config import DatasetConfigs

def download_file(url: str, output_path: str, name: str):
    logging.info(f"Downloading {name} dataset")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    download_progressbar_train = TQDMConfigs().download_progressbar_color
    with open(output_path, "wb") as file, tqdm(
        desc=output_path,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        colour=download_progressbar_train,
    ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))
        bar.close()


def extract_tar_gz(file_path: str, extract_path: str, name: str):
    logging.info(f"Extract .tar.gz archive {name} dataset")
    if tarfile.is_tarfile(file_path):
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_path)
        logging.info(f"Extracted to: {extract_path}")
    else:
        logging.info("Not a valid tar.gz file.")


if __name__ == "__main__":
    dataset_configs = DatasetConfigs().get_config()
    vivos_ds = dataset_configs['vivos']
    vlsp_ds = dataset_configs['vlsp']
    print(dataset_configs)
    os.makedirs(vivos_ds['data_dir'], exist_ok=True)
    download_file(vivos_ds['url'], vivos_ds['download_file'], vivos_ds['name'])
    extract_tar_gz(vivos_ds['download_file'], vivos_ds['data_dir'], vivos_ds['name'])

    os.makedirs(vlsp_ds['data_dir'], exist_ok=True)
    download_file(vlsp_ds['url'], vlsp_ds['download_file'], vlsp_ds['name'])
    extract_tar_gz(vlsp_ds['download_file'], vlsp_ds['data_dir'], vlsp_ds['name'])
