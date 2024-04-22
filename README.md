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

**Neural Network Regression with Extreme Inputs**
- files in `experiments/neural_network_regression/`
- configure and run using the `run_nn_experiment.ipynb` notebook
- results analysed with `nn_results_analysis.ipynb`

**Density Estimation with Synthetic Data**
- files in `experiments/neural_network_regression/density_estimation_heavy_tailed_nuisance/`
- generate synthetic datasets with `synth_de_data_generation.ipynb`
- configure and run experiments using the `run_synth_de_experiment.ipynb` notebook (outputs csv)
- analyse results with `synthetic_analysis.ipynb`
- `de_shift_experiments.ipynb` notebook contains some inspections of the fitted densities

