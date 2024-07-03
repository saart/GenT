import os.path
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Dict

from drivers.netshare.netshare_driver import NetShareDriver
from ml.app_utils import GenTConfig


CONFIGS = [
    NetShareDriver(GenTConfig(chain_length=chain_length))
    for chain_length in [2, 3, 4]
]

def _get_raw_data_size(config: NetShareDriver) -> int:
    return sum(
        f.stat().st_size
        for f in Path(config.gen_t_config.get_raw_normalized_data_dir()).glob(
            "**/*"
        )
        if f.is_file()
    )


def get_raw_data_size() -> Dict[int, int]:
    return {config.gen_t_config.chain_length: _get_raw_data_size(config) for config in CONFIGS}


def _get_raw_data_gzip_size(config: NetShareDriver) -> int:
    target_file = f"{tempfile.mkdtemp()}/model.tar.gz"
    tar = tarfile.open(target_file, "w:gz")
    tar.add(config.gen_t_config.get_raw_normalized_data_dir(), arcname="model")
    tar.close()
    stat = Path(target_file).stat().st_size
    shutil.rmtree(target_file)
    return stat


def get_raw_data_gzip_size() -> Dict[int, int]:
    return {config.gen_t_config.chain_length: _get_raw_data_gzip_size(config) for config in CONFIGS}


if __name__ == '__main__':
    print("get_raw_data_size", get_raw_data_size())
    print("get_raw_data_gzip_size", get_raw_data_gzip_size())

