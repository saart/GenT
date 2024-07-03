import contextlib
import importlib
import multiprocessing
import os.path
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Iterable

from drivers.base_driver import BaseDriver
from drivers.gent.data import ALL_TRACES
from drivers.gent.gent_driver import GenTDriver
from fidelity.raw_sql import fill_benchmark
from ml.app_utils import GenTConfig, clear, GenTBaseConfig
from paper.ops_utils import FidelityResult, load_results, store_and_upload_results


def measure_configuration(
    driver: BaseDriver,
    lock: Optional[multiprocessing.synchronize.Lock] = None,
    skip_if_exists: bool = True,
) -> FidelityResult:
    if skip_if_exists and driver.get_results_key() in load_results():
        existing_result = load_results()[driver.get_results_key()]
        print(f"Already processed {driver.get_results_key()}")
        return existing_result

    print("Starting", driver.get_driver_name(), driver.gen_t_config.to_string())

    driver.train_and_generate()
    result = FidelityResult(
        model_size=driver.get_model_size(),
        gzip_model_size=driver.get_model_gzip_size(),
    )
    print(f"Driver: {driver}, Result: {result}")
    with lock or contextlib.suppress():
        store_and_upload_results(driver.gen_t_config, result, driver.get_driver_name())
    if "MacBook-Pro" not in os.uname().nodename:
        driver.upload_and_clear(True)
        clear()
    return result


def measure_multiple_configurations(drivers: Iterable[BaseDriver], max_workers=1) -> None:
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        lock = multiprocessing.Manager().Lock()
        futures = [
            pool.submit(measure_configuration, driver, lock, True) for driver in drivers
        ]
        result = [f.result() for f in futures]
    print(result)


def iterations_exp():
    print("GenT iterations (time-based)")
    configs = [
        GenTConfig(chain_length=2, tx_start=0, tx_end=tx_count, iterations=iterations)
        for tx_count in [1_000]#, 2_000, 5_000, 10_000, 15_000]
        for iterations in [1, 2, 3, 4, 5, 6, 7, 10, 20, 30]
    ]
    for config in configs:
        print(f"#### iterations {config.iterations} tx_count {config.tx_end} #####")
        measure_configuration(GenTDriver(config), skip_if_exists=True)


def batch_size():
    print("GenT changing CTGAN's generator dimension")
    measure_multiple_configurations(map(GenTDriver, [
        GenTConfig(chain_length=2, tx_end=10_000),
        GenTConfig(chain_length=2, tx_end=15_000),
        GenTConfig(chain_length=2, tx_end=20_000),
    ]), max_workers=3)


def simple_ablations():
    print("GenT simple_ablations")
    configs = [
        GenTConfig(chain_length=2, tx_start=0, tx_end=ALL_TRACES, iterations=10, independent_chains=True),
        GenTConfig(chain_length=2, tx_start=0, tx_end=ALL_TRACES, iterations=10, with_gcn=False),
        GenTConfig(chain_length=2, tx_start=0, tx_end=ALL_TRACES, iterations=10, start_time_with_metadata=True),
    ]
    for i, config in enumerate(configs):
        print(f"####### simple_ablations index: {i} ########")
        measure_configuration(GenTDriver(config), skip_if_exists=False)


def ctgan_dim():
    print("GenT ctgan_dim")
    configs = [
        GenTConfig(chain_length=2, tx_start=0, tx_end=ALL_TRACES, iterations=10, generator_dim=(128,)),
        GenTConfig(chain_length=2, tx_start=0, tx_end=ALL_TRACES, iterations=10, generator_dim=(128, 128)),
        GenTConfig(chain_length=2, tx_start=0, tx_end=ALL_TRACES, iterations=10, generator_dim=(256, 256)),
        GenTConfig(chain_length=2, tx_start=0, tx_end=ALL_TRACES, iterations=10, generator_dim=(256,)),
    ]
    for config in configs:
        print("#### ctgan_dim #####", config.generator_dim)
        measure_configuration(GenTDriver(config), skip_if_exists=True)


def chain_length():
    print("GenT chain length")
    configs = [
        GenTConfig(chain_length=2, tx_start=0, tx_end=ALL_TRACES, iterations=10),
        GenTConfig(chain_length=3, tx_start=0, tx_end=ALL_TRACES, iterations=10),
        GenTConfig(chain_length=4, tx_start=0, tx_end=ALL_TRACES, iterations=10),
        GenTConfig(chain_length=5, tx_start=0, tx_end=ALL_TRACES, iterations=10),
    ]
    for config in configs:
        print("#### Chain length #####", config.chain_length)
        measure_configuration(GenTDriver(config), skip_if_exists=True)


def main(arg: Optional[str] = None) -> None:
    arg = arg or sys.argv[1]
    if arg == "chain_length":
        chain_length()
    elif arg == "ctgan_dim":
        ctgan_dim()
    elif arg == "iterations":
        iterations_exp()
    elif arg == "simple_ablations":
        simple_ablations()
    elif arg == "batch_size":
        batch_size()
    else:
        raise ValueError("Unknown experiment")


if __name__ == "__main__":
    iterations = 3
    for desc in os.listdir("/Users/saart/cmu/GenT/traces"):
        if desc == "wildryde":
            continue
        for i in range(iterations):
            traces_dir = f"/Users/saart/cmu/GenT/traces/{desc}"
            driver = GenTDriver(GenTConfig(chain_length=2, iterations=30, tx_end=10000 + i, traces_dir=traces_dir))
            driver.train_and_generate()
            fill_benchmark(real_data_dir=traces_dir, syn_data_dir=driver.get_generated_data_folder(), desc=desc, variant=i)
    # driver = TabFormerDriver(GenTBaseConfig(chain_length=3, iterations=5))
    # measure_configuration(driver, skip_if_exists=False)
    # iterations_exp()
    # chain_length()
    # ctgan_dim()
    # simple_ablations()
