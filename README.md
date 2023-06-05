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
To run a specific experiment, first configure by editing the target script, e.g. `experiments/run_variational_inference.py`, and then execute.
```
python experiments/run_variational_inference.py
```

Experiment outputs are stored in `experiment_output/`.
This includes training data (loss paths) and the final model.
The results are analysed in e.g. `experiments/neals_funnel_analysis.ipynb`.

To use the posterior DB experimetns set the `POSTERIOR_DB_PATH` to the location of the posterior DB files.