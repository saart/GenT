import boto3
import os
import shutil
import tarfile
import tempfile
import time
from typing import List

from multiprocessing.pool import Pool

from drivers.base_driver import BaseDriver, DriverType
from drivers.gent.data import ALL_TRACES
from drivers.gent.metadata_generator_ctgan import MetadataGenerator, \
    train_and_save_root, train_and_save_chained, continue_train_and_save_root, continue_train_and_save_chained
from drivers.gent.start_time_generator_ctgan import StartTimesGenerator, \
    train_and_save as train_and_save_start_time, continue_train_and_save as continue_train_and_save_start_time
from ml.app_utils import GenTConfig


class GenTDriver(BaseDriver):
    def __init__(self, gen_t_config: GenTConfig):
        self.gen_t_config: GenTConfig
        super().__init__(gen_t_config)
        self.metadata_generator = None
        self.start_time_generator = None

    def get_driver_name(self) -> DriverType:
        return "genT"

    def pretty_name(self) -> str:
        return "GenT"

    def get_normalized_generated_data_folder(self):
        return os.path.join(self.get_work_folder(), "generated")

    def _get_model_files(self) -> List[str]:
        return [
            os.path.join(self.get_work_folder(), "metadata", "chained_ctgan_generator.pkl"),
            os.path.join(self.get_work_folder(), "metadata", "root_ctgan_generator.pkl"),
            os.path.join(self.get_work_folder(), "metadata", "column_to_values.pkl"),
            os.path.join(self.get_work_folder(), "metadata", "graph_index_to_chains.pkl"),
            os.path.join(self.get_work_folder(), "metadata", "graph_index_to_edges.pkl"),
            os.path.join(self.get_work_folder(), "metadata", "node_to_index.pkl"),
            os.path.join(self.get_work_folder(), "metadata", "best_root_seed.pkl"),
            os.path.join(self.get_work_folder(), "metadata", "best_chained_seed.pkl"),
            os.path.join(self.get_work_folder(), "start_time", "start_time_ctgan_generator.pkl"),
            os.path.join(self.get_work_folder(), "start_time", "best_seed.pkl"),
            os.path.join(self.get_work_folder(), "start_time", "graph_values.pkl"),
            os.path.join(self.get_work_folder(), "start_time", "min_real_timestamp.pkl"),
            os.path.join(self.get_work_folder(), "start_time", "max_real_timestamp.pkl"),
        ]

    def get_model_size(self) -> int:
        return sum(os.path.getsize(f) for f in self._get_model_files())

    def get_model_gzip_file(self) -> str:
        print("Zipping model files")
        target_file = f"{tempfile.mkdtemp()}/model.tar.gz"
        tar = tarfile.open(target_file, "w:gz")
        for f in self._get_model_files():
            tar.add(f)
        tar.close()
        return target_file

    def download(self):
        s3_key = f"{self.get_driver_name()}/{self.gen_t_config.to_string()}.tar.gz"
        local_path = os.path.join(self.get_work_folder(), "downloaded.tar.gz")

        print("Downloading model files from s3")
        os.makedirs(self.get_work_folder(), exist_ok=True)
        s3 = boto3.session.Session(profile_name="gent").resource('s3')
        s3.Bucket("gent-results").download_file(s3_key, local_path)

        print("Unzipping model files")
        tar = tarfile.open(local_path, "r:gz")
        for file in tar.getmembers():
            subdir, file_name = file.name.split("/")[-2:]
            os.makedirs(os.path.join(self.get_work_folder(), subdir), exist_ok=True)
            file.name = file_name
            tar.extract(file, os.path.join(self.get_work_folder(), subdir))
        tar.close()

    def train_and_generate(self) -> None:
        shutil.rmtree(self.get_work_folder(), ignore_errors=True)
        start = time.time()
        # train_and_save_start_time(self.gen_t_config, os.path.join(self.get_work_folder(), "start_time"))
        # train_and_save_root(self.gen_t_config, os.path.join(self.get_work_folder(), "metadata"))
        # train_and_save_chained(self.gen_t_config, os.path.join(self.get_work_folder(), "metadata"))
        with Pool(processes=3) as pool:
            processes = [
                pool.apply_async(train_and_save_start_time, (self.gen_t_config, os.path.join(self.get_work_folder(), "start_time"))),
                pool.apply_async(train_and_save_root, (self.gen_t_config, os.path.join(self.get_work_folder(), "metadata"))),
                pool.apply_async(train_and_save_chained, (self.gen_t_config, os.path.join(self.get_work_folder(), "metadata"))),
            ]
            [p.get() for p in processes]  # raise exceptions if any
            pool.close()
            pool.join()
        print(f"Training took {time.time() - start} seconds")
        self.generate()

    def generate(self, param: int = 0, from_downloaded: bool = False, suffix: str = '') -> None:
        start_time_generator = self.get_start_time_generator()
        metadata_generator = self.get_metadata_generator()

        start = time.time()
        ts_corpus = start_time_generator.generate_timestamps_corpus()
        print("Generated start times")
        metadata_generator.generate_traces_corpus(
            target_dir_path=self.get_generated_data_folder() + suffix,
            ts_corpus=ts_corpus,
        )
        print(f"Full Generation took {time.time() - start} seconds into {self.get_generated_data_folder()}")

    def get_start_time_generator(self) -> StartTimesGenerator:
        if not self.start_time_generator:
            self.start_time_generator = StartTimesGenerator.get(self.gen_t_config)
            self.start_time_generator.load(path=os.path.join(self.get_work_folder(), "start_time"))
        return self.start_time_generator

    def get_metadata_generator(self) -> MetadataGenerator:
        if not self.metadata_generator:
            self.metadata_generator = MetadataGenerator.get(self.gen_t_config)
            self.metadata_generator.load(path=os.path.join(self.get_work_folder(), "metadata"))
        return self.metadata_generator

    def store_metric(self, metric_name: str, value: float) -> None:
        open(os.path.join(self.get_work_folder(), f"{metric_name}.txt"), "w").write(str(value))

    def get_metric(self, metric_name: str) -> float:
        path = os.path.join(self.get_work_folder(), f"{metric_name}.txt")
        if not os.path.exists(path):
            raise Exception(f"Metric {metric_name} not found")
        return float(open(path, "r").read())


if __name__ == "__main__":
    driver = GenTDriver(GenTConfig(iterations=5, tx_end=ALL_TRACES))
    # driver.train_and_generate()
    driver.roll()
    # start_time_generator = StartTimesGenerator.get(driver.gen_t_config)
    # start_time_generator.load(path=os.path.join(driver.get_work_folder(), "start_time"))
    # val = start_time_generator.compare()[0][0]
    # driver.store_metric("start_time", val)
