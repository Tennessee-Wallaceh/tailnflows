# tailnflows
Repository for experiments related to "flexible tails for normalising flows".

## Structure

The flow models are defined in `tailnflows.models.flows`.
To allow easily testing different configurations the models are defined according to the structure specificied in the `tailnflows.models.flows.ExperimentFlow` model.
This allows changing the base distribution, rotation, final transformation and an arbitrary sequence of normalizing flow transformations.

The specific models are then created by calling 
```python
build_{model_name}(
    dim: int, # dimension of problem
    use: ModelUse, # either for density estimation or variational inference 
    base_transformation_init: Callable[[int], list[Transform]], # produce the sequence of transformations in data->noise direction
    constraint_transformation: Optional[Transform] = None,
    model_kwargs: ModelKwargs, # any model specific config
    nn_kwargs: NNKwargs = {}, # any model specific neural net config
)
```
See `tailnflows.models.flows.build_base_model` for the basic usage.

The base transformations used are defined as functions (named as `base_{name}_transformation`) which produce a list of transformations. This may seem complicated, but ensures that transformations are consistent between training runs.

There are a number of options related to configuration of the flows.
More details and rationale for these choices can be found in the paper.

## environment

The code is developed on python 3.9.
Setup the environment and install the package from root of directory with:
```
pip install -r requirements.txt
pip install -e . # editable install for development
```

## running experiments
Scripts for configuring and running experiments are in `experiments/`.

**Neural Network Regression with Extreme Inputs**
- files in `experiments/neural_network_regression/`
- configure by editing `configured_experiments` in `experiments/neural_network_regression/run_nn_experiment.py`
- run using `python run_nn_experiment.py`
- results analysed with `nn_results_analysis.ipynb`

**Density Estimation with Synthetic Data**
- files in `experiments/density_estimation_synthetic/`
- generate synthetic datasets with `synth_de_data_generation.ipynb`
- configure and run experiments using the `python run_synth_de_experiment.py` script (edit the `configured_experiments` function)
- analyse results with `synthetic_analysis.ipynb`
- `de_shift_experiments.ipynb` notebook contains some inspections of the fitted densities



