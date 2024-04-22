import pickle
from pathlib import Path
from typing import Any
from tailnflows.utils import get_project_root


def add_raw_data(path: str, label: str, data: Any) -> None:
    rd_path = f"{get_project_root()}/experiment_output/{path}.p"

    data_file = Path(rd_path)
    if not data_file.is_file():
        pickle.dump({}, open(rd_path, "wb"))
    else:
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
