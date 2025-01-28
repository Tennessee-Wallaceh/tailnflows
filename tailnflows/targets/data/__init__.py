from .sp500_returns import load_return_data
from .fama5 import load_data as fama5_load
from .insurance import load_data as insurance_load

real_data_sources = {
    'sp500': lambda: load_return_data(300)[0], # just the data
    'fama5': fama5_load,
    'insurance': insurance_load,
}