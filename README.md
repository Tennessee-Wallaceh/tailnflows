# tailnflows
Repository for experiments related to "flexible tails for normalising flows"

## environment
A conda (https://docs.conda.io/en/latest/miniconda.html) environment file is provided.
Setup the environment and install the package from root of directory with:
```
conda env create --file environment.yml
conda activate tailnflows
pip install -e . # editable install for development
```

## run experiments
Scripts for configuring and running experiments are in `experiments/`.
The configuration has a few different options, so I manage it in python.
To run a specific experiment execute the following, inspect the script for other options.
```
python experiments/run_density_estimation.py -model_name {model_name} -target_config top_20
```

Experiment outputs (final model state and training data) are stored in `experiment_output/`.
This includes training data (loss paths) and the final model.
The results are analysed in e.g. `experiments/fret_analysis.ipynb`.

<!-- To use the posterior DB experimetns set the `POSTERIOR_DB_PATH` to the location of the posterior DB files. -->