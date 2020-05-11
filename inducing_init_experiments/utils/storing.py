import os
import pickle
import re
from glob import glob


def get_next_filename(path, base_filename="data", extension="pkl"):
    if not os.path.exists(path):
        os.makedirs(path)
    largest_existing_number = max([int(re.findall(r'\d+', fn)[-1]) for fn in glob(f"{path}/{base_filename}*")] + [0])
    path = f"{path}/{base_filename}{largest_existing_number + 1}"
    if extension is not None:
        path = f"{path}.{extension}"
    return path


def store_pickle(data, base_path, base_filename="data"):
    with open(get_next_filename(base_path, base_filename), 'wb') as outfile:
        pickle.dump(data, outfile)


def load_existing_runs(path, base_filename="data"):
    existing_runs = []
    for fn in glob(f"{path}/{base_filename}*"):
        with open(fn, 'rb') as fp:
            existing_runs.append((pickle.load(fp), fn))
    return existing_runs


def weak_dictionary_compare(source_dict, target_dict):
    """
    Returns `True` if all values that are present in `source_dict`, are the
    same as the values of the same key in `target_dict.
    """



def find_run(existing_runs, run_details):
    return [run for run in existing_runs if run["run_details"] == run_details]
