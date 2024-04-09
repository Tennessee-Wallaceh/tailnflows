import os
from pathlib import Path
import sys
import subprocess

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
    "git+https://github.com/Tennessee-Wallaceh/nflows",
    "git+https://github.com/Tennessee-Wallaceh/marginalTailAdaptiveFlow.git",
]


def get_project_root():
    if IN_COLAB:
        from google.colab import drive

        drive.mount("/content/drive")
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
