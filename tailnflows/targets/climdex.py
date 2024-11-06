import xarray as xr
from tailnflows.utils import get_project_root
import numpy as np
import torch
from marginal_tail_adaptive_flows.utils.weather_helper import (
    compute_layer_cloud_optical_depth,
    to_normalized_dataset,
    train_test_split_dataset,
    to_stacked_array,
)


def load_climdex_data():
    """
    This is replicated from the marginalTailAdaptiveFlow paper
    https://github.com/MikeLasz/marginalTailAdaptiveFlow/blob/master/utils/flows.py#L532
    """
    columns = slice(0, None)
    ds_inputs = xr.open_dataset(f"{get_project_root()}/data/nwp_saf_profiles_in.nc")
    ds_inputs = compute_layer_cloud_optical_depth(ds_inputs)
    inputs_relevant = ["temperature_fl", "pressure_hl"]
    inputs_relevant += ["layer_cloud_optical_depth"]
    ds_true_in = ds_inputs[inputs_relevant].sel(column=columns)

    ds_true, stats_info = to_normalized_dataset(ds_true_in)

    ds_train, ds_test = train_test_split_dataset(
        ds_true, test_size=0.6, dim="column", shuffle=True, seed=42
    )
    ds_test, ds_validation = train_test_split_dataset(
        ds_test, test_size=0.33334, dim="column", shuffle=True, seed=42
    )
    ds_train, _ = train_test_split_dataset(
        ds_train, train_size=1, dim="column", shuffle=False
    )

    ds_true_in["pressure_hl"] /= 100  # Convert Pa to hPa

    col_name = "pressure_hl"
    # impute a tiny bit of noise for
    # atmospheric pressure below 80:
    noise_sd = 0.001
    impute_noise = True
    ds_pressure_train = ds_train[col_name].to_numpy()
    ds_pressure_train[:, :80] += impute_noise * np.random.normal(
        0, noise_sd, [len(ds_pressure_train), 80]
    )
    ds_train[col_name].data = ds_pressure_train
    ds_pressure_test = ds_test[col_name].to_numpy()
    ds_pressure_test[:, :80] += impute_noise * np.random.normal(
        0, noise_sd, [len(ds_pressure_test), 80]
    )
    ds_test[col_name].data = ds_pressure_test
    ds_pressure_val = ds_validation[col_name].to_numpy()
    ds_pressure_val[:, :80] += impute_noise * np.random.normal(
        0, noise_sd, [len(ds_pressure_val), 80]
    )
    ds_validation[col_name].data = ds_pressure_val

    col_name = "layer_cloud_optical_depth"
    ds_opt_train = ds_train[col_name].to_numpy()
    ds_opt_train[:, :80] += impute_noise * np.random.normal(
        0, noise_sd, [len(ds_opt_train), 80]
    )
    ds_train[col_name].data = ds_opt_train
    ds_opt_test = ds_test[col_name].to_numpy()
    ds_opt_test[:, :80] += impute_noise * np.random.normal(
        0, noise_sd, [len(ds_opt_test), 80]
    )
    ds_test[col_name].data = ds_opt_test
    ds_opt_val = ds_validation[col_name].to_numpy()
    ds_opt_val[:, :80] += impute_noise * np.random.normal(
        0, noise_sd, [len(ds_opt_val), 80]
    )
    ds_validation[col_name].data = ds_opt_val

    data_train = torch.tensor(to_stacked_array(ds_train)[0].to_numpy())
    data_test = torch.tensor(to_stacked_array(ds_test)[0].to_numpy())
    data_val = torch.tensor(to_stacked_array(ds_validation)[0].to_numpy())

    return data_train, data_val, data_test
