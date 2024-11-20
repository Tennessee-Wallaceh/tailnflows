from tailnflows.utils import get_project_root, get_data_path
import pandas as pd
import numpy as np
import torch


def load_data():
    """
    Replicated from https://github.com/feynmanliang/ftvi/blob/main/figures/fat-tailed-insurance-inpatient-outpatient.ipynb
    """
    df_in = pd.read_csv(
        f"{get_data_path()}/insurance/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv"
    )
    df_out = pd.read_csv(
        f"{get_data_path()}/insurance/DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv"
    )
    df_claims = (
        df_in[["CLM_PMT_AMT", "CLM_FROM_DT"]]
        .groupby("CLM_FROM_DT")
        .sum()
        .join(
            df_out[["CLM_PMT_AMT", "CLM_FROM_DT"]].groupby("CLM_FROM_DT").sum(),
            rsuffix="_out",
            how="inner",
        )
        .rename(
            columns={
                "CLM_PMT_AMT": "total_inpatient",
                "CLM_PMT_AMT_out": "total_outpatient",
            }
        )
    )

    for sub in ["inpatient", "outpatient"]:
        df_claims[f"log_total_{sub}"] = np.log(df_claims[f"total_{sub}"])
    df_claims = df_claims[[col for col in df_claims.columns if col.startswith("log")]]

    return torch.tensor(df_claims.to_numpy())
