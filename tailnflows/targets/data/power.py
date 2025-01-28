from tailnflows.utils import get_data_path
import pandas as pd
import torch
import numpy as np


def load_data():
    # Global_reactive_power discarded see 
    # D.2 in https://arxiv.org/pdf/1705.07057 for discussion
    numeric_columns = [
        'Global_active_power',
        # 'Global_reactive_power',  
        'Voltage', 
        'Global_intensity',
        'Sub_metering_1', 
        'Sub_metering_2', 
        'Sub_metering_3'
    ]
    df = pd.read_csv(
        f"{get_data_path()}/power/data.csv",
    )[numeric_columns]

    return torch.tensor(df.to_numpy())

def load_power():
    def load_data():
        df = pd.read_csv(
            f"{get_data_path()}/power/data.csv",
        )

        x_np = df.to_numpy()

        rng = np.random.RandomState(42)
        rng.shuffle(x_np)
        N = data.shape[0]

    # def load_data_split_with_noise():
    #     rng = np.random.RandomState(42)

    #     data = load_data()
    #     rng.shuffle(data)
    #     N = data.shape[0]

    #     data = np.delete(data, 3, axis=1)
    #     data = np.delete(data, 1, axis=1)
    #     ############################
    #     # Add noise
    #     ############################
    #     # global_intensity_noise = 0.1*rng.rand(N, 1)
    #     voltage_noise = 0.01 * rng.rand(N, 1)
    #     # grp_noise = 0.001*rng.rand(N, 1)
    #     gap_noise = 0.001 * rng.rand(N, 1)
    #     sm_noise = rng.rand(N, 3)
    #     time_noise = np.zeros((N, 1))
    #     # noise = np.hstack((gap_noise, grp_noise, voltage_noise, global_intensity_noise, sm_noise, time_noise))
    #     # noise = np.hstack((gap_noise, grp_noise, voltage_noise, sm_noise, time_noise))
    #     noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
    #     data += noise

    #     N_test = int(0.1 * data.shape[0])
    #     data_test = data[-N_test:]
    #     data = data[0:-N_test]
    #     N_validate = int(0.1 * data.shape[0])
    #     data_validate = data[-N_validate:]
    #     data_train = data[0:-N_validate]

    #     return data_train, data_validate, data_test

    # def load_data_normalised():
    #     data_train, data_validate, data_test = load_data_split_with_noise()
    #     data = np.vstack((data_train, data_validate))
    #     mu = data.mean(axis=0)
    #     s = data.std(axis=0)
    #     data_train = (data_train - mu) / s
    #     data_validate = (data_validate - mu) / s
    #     data_test = (data_test - mu) / s

    #     return data_train, data_validate, data_test

    # return load_data_normalised()