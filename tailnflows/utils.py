import os
from pathlib import Path
import sys
import subprocess
import pickle
from typing import Any
import torch
import multiprocessing as mp

IN_COLAB = "google.colab" in sys.modules

pip_dependencies = [
    "jupyter",
    "matplotlib",
    "numpy",
    "scipy",
    "tqdm",
    "pandas",
    "seaborn",
    "posteriordb",
    "jaxtyping",
    "git+https://github.com/Tennessee-Wallaceh/nflows",
    "git+https://github.com/Tennessee-Wallaceh/marginalTailAdaptiveFlow.git",
]


def get_project_root():
    if IN_COLAB:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=True)
        # this needs to correspond to actual location in google drive, populated with data
        os.environ["TAILNFLOWS_HOME"] = "/content/drive/MyDrive/tailnflows"

    root_path = os.environ.get("TAILNFLOWS_HOME", Path(__file__).parent.parent)

    return root_path


def get_data_path():
    return f"{get_project_root()}/data"


def get_experiment_output_path():
    return f"{get_project_root()}/experiment_output"


def configure_colab_env():
    for requirement in pip_dependencies:
        result = subprocess.run(["pip", "install", requirement], capture_output=True)
        if len(result.stderr) > 0:
            print(result.stderr)

    print("Setup complete!")


def add_raw_data(path: str, label: str, data: Any, force_write: bool = False) -> None:
    rd_path = f"{get_project_root()}/experiment_output/{path}.p"

    data_file = Path(rd_path)

    if not data_file.is_file():
        data_file.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump({}, open(rd_path, "wb"))
    elif not force_write:
        confirm = input("Experiment data already present, reset? (y/n)")
        if confirm == "y":
            pickle.dump({}, open(rd_path, "wb"))
        else:
            print("no reset, appending data...")

    raw_data = pickle.load(open(rd_path, "rb"))
    if label not in raw_data:
        raw_data[label] = []
    raw_data[label].append(data)
    pickle.dump(raw_data, open(rd_path, "wb"))


def load_raw_data(path: str) -> Any:
    rd_path = f"{get_project_root()}/experiment_output/{path}.p"
    return pickle.load(open(rd_path, "rb"))


def load_torch_data(path: str) -> Any:
    rd_path = f"{get_project_root()}/data/{path}.p"
    if not torch.cuda.is_available():
        data = torch.load(open(rd_path, "rb"), map_location=torch.device("cpu"))
    else:
        data = torch.load(open(rd_path, "rb"), map_location=torch.device("cpu"))

    return data

class RunWrapper:
    def __init__(self, run_experiment):
        self.run_experiment = run_experiment
    def __call__(self, exp_ix_kwargs):
        exp_ix, kwargs = exp_ix_kwargs
        self.run_experiment(experiment_ix=exp_ix+ 1, **kwargs)


def parallel_runner(run_experiment, experiments, max_runs=3):
    print(f"{len(experiments)} experiments to run...")
    with mp.Pool(max_runs) as p:
        p.map(RunWrapper(run_experiment), list(enumerate(experiments)), chunksize=1)
