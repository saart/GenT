"""
This test train 5K traces on normal data,
then roll the model to train on 5K traces on higher errors,
then roll the model to train on 5K traces on higher traces rate,
then roll the model and train on 5K traces on less errors
"""
from multiprocessing import Pool
from pathlib import Path

from drivers.gent.data import CONFIG_BASE, CONFIG_ERROR, CONFIG_RATE, CONFIG_LESS_ERRORS
from drivers.gent.metadata_generator_ctgan import MetadataGenerator, \
    train_and_save_root, train_and_save_chained, continue_train_and_save_root, continue_train_and_save_chained
from drivers.gent.start_time_generator_ctgan import StartTimesGenerator, \
    train_and_save as train_and_save_start_time

BASE_PATH = Path('.').absolute() / "adaption_exp"
ROLLING_PATH = BASE_PATH / "rolling"
NON_ROLLING_PATH = BASE_PATH / "non_rolling"


def start_time(is_rolling: bool):
    path = ROLLING_PATH if is_rolling else NON_ROLLING_PATH
    train_and_save_start_time(CONFIG_BASE, path / "base" / "start_time", is_roll=True)
    train_and_save_start_time(CONFIG_ERROR, path / "error" / "start_time", is_roll=True)
    train_and_save_start_time(CONFIG_RATE, path / "rate" / "start_time", is_roll=True)
    train_and_save_start_time(CONFIG_LESS_ERRORS, path / "less_errors" / "start_time", is_roll=True)


def root_metadata(is_rolling: bool):
    if is_rolling:
        train_and_save_root(CONFIG_BASE, ROLLING_PATH / "base" / "metadata", is_roll=True)
        continue_train_and_save_root(CONFIG_ERROR, ROLLING_PATH / "error" / "metadata", from_path=ROLLING_PATH / "base" / "metadata")
        continue_train_and_save_root(CONFIG_RATE, ROLLING_PATH / "rate" / "metadata", from_path=ROLLING_PATH / "error" / "metadata")
        continue_train_and_save_root(CONFIG_LESS_ERRORS, ROLLING_PATH / "less_errors" / "metadata", from_path=ROLLING_PATH / "rate" / "metadata")
    else:
        # Don't continue
        train_and_save_root(CONFIG_BASE, NON_ROLLING_PATH / "base" / "metadata", is_roll=True)
        train_and_save_root(CONFIG_ERROR, NON_ROLLING_PATH / "error" / "metadata", is_roll=True)
        train_and_save_root(CONFIG_RATE, NON_ROLLING_PATH / "rate" / "metadata", is_roll=True)
        train_and_save_root(CONFIG_LESS_ERRORS, NON_ROLLING_PATH / "less_errors" / "metadata", is_roll=True)

def chained_metadata(is_rolling: bool):
    if is_rolling:
        train_and_save_chained(CONFIG_BASE, ROLLING_PATH / "base" / "metadata", is_roll=True)
        continue_train_and_save_chained(CONFIG_ERROR, ROLLING_PATH / "error" / "metadata", from_path=ROLLING_PATH / "base" / "metadata")
        continue_train_and_save_chained(CONFIG_RATE, ROLLING_PATH / "rate" / "metadata", from_path=ROLLING_PATH / "error" / "metadata")
        continue_train_and_save_chained(CONFIG_LESS_ERRORS, ROLLING_PATH / "less_errors" / "metadata", from_path=ROLLING_PATH / "rate" / "metadata")
    else:
        # Don't continue
        train_and_save_chained(CONFIG_BASE, NON_ROLLING_PATH / "base" / "metadata", is_roll=True)
        train_and_save_chained(CONFIG_ERROR, NON_ROLLING_PATH / "error" / "metadata", is_roll=True)
        train_and_save_chained(CONFIG_RATE, NON_ROLLING_PATH / "rate" / "metadata", is_roll=True)
        train_and_save_chained(CONFIG_LESS_ERRORS, NON_ROLLING_PATH / "less_errors" / "metadata", is_roll=True)


def generate(is_rolling: bool, exp_name: str):
    path = ROLLING_PATH if is_rolling else NON_ROLLING_PATH
    start_time_generator = StartTimesGenerator.get(CONFIG_BASE).load_all(path=path / exp_name / "start_time")
    metadata_generator = MetadataGenerator.get(CONFIG_BASE).load_all(path=path / exp_name / "metadata")

    ts_corpus = start_time_generator.generate_timestamps_corpus()
    metadata_generator.generate_traces_corpus(
        target_dir_path=path / exp_name / "generated",
        ts_corpus=ts_corpus,
    )


def main():
    generators = [start_time, root_metadata, chained_metadata]
    is_rollings = [True]
    with Pool(processes=3) as pool:
        processes = [pool.apply_async(gen, (is_rolling,)) for gen in generators for is_rolling in is_rollings]
        [p.get() for p in processes]
        # [gen() for gen in generators]

    cases = ["base", "error", "rate", "less_errors"]
    with Pool(processes=4) as pool:
        processes = [pool.apply_async(generate, (is_rolling, c,)) for c in cases for is_rolling in is_rollings]
        [p.get() for p in processes]
        # [generate(c) for c in cases]


if __name__ == '__main__':
    main()
