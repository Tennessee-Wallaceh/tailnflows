"""
An attempt to systemize data locations, not guaranteed to be stable
"""
from pathlib import Path
from tailnflows.utils import get_data_path
import argparse

def download_fama5():
    import subprocess

    try:
        subprocess.run(["pip", "install", "pandas_datareader"], check=True)
        # Run your code here
        import pandas_datareader as pdr

        ds = pdr.data.DataReader('5_Industry_Portfolios_daily', 'famafrench')
        data_path = f'{get_data_path()}/fama5/data.csv'
        data_file = Path(data_path)

        if not data_file.is_file():
            data_file.parent.mkdir(parents=True, exist_ok=True)

        ds[0].to_csv(data_path)

    finally:
        subprocess.run(["pip", "uninstall", "pandas_datareader", "-y"], check=True)

def download_power():
    import subprocess

    try:
        subprocess.run(["pip", "install", "ucimlrepo"], check=True)
        # Run your code here
        from ucimlrepo import fetch_ucirepo 
  
        # fetch dataset 
        individual_household_electric_power_consumption = fetch_ucirepo(id=235) 
        
        # data (as pandas dataframes) 
        X = individual_household_electric_power_consumption.data.features 
        y = individual_household_electric_power_consumption.data.targets
        
        data_path = f'{get_data_path()}/power/data.csv'
        data_file = Path(data_path)

        if not data_file.is_file():
            data_file.parent.mkdir(parents=True, exist_ok=True)

        X.to_csv(data_path)

    finally:
        subprocess.run(["pip", "uninstall", "ucimlrepo", "-y"], check=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data download helper.')
    
    # Both fama5 and power are booleans toggled by presence of the flag.
    parser.add_argument('--fama5', action='store_true', help='Attempt fama5 download')
    parser.add_argument('--power', action='store_true', help='Attempt power download')
    
    args = parser.parse_args()
    
    # Check the values of the flags
    if args.fama5:
        download_fama5()
    
    if args.power:
        download_power()