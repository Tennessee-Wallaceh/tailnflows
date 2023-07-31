
import warnings
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from marginal_tail_adaptive_flows.utils.tail_estimation import make_plots

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, *args, **kwargs):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def estimate_df(marginal_data):
    ordered_data = np.sort(np.abs(marginal_data))[::-1]

    amse_border = 1.

    eps_stop = 1 - float(len(ordered_data[np.where(ordered_data <= amse_border)])) / len(ordered_data)
    plt.interactive(False)

    with warnings.catch_warnings():
        with HiddenPrints():
            warnings.simplefilter("ignore")
            tail_estimators = make_plots(
                ordered_data, 
                output_file_path='tst', 
                number_of_bins=30,
                r_smooth=2, 
                alpha=0.6, 
                hsteps=200, 
                bootstrap_flag=1, 
                t_bootstrap=0.5,
                r_bootstrap=500, 
                diagn_plots=False, 
                eps_stop=eps_stop, 
                theta1=0.01, 
                theta2=0.99,
                verbose=False, 
                noise_flag=0, 
                p_noise=1, 
                savedata=False
            )

    if tail_estimators["moments"] == tail_estimators["kernel"] == 0:
        final_estimator = 0
        print("This distribution is light-tailed!")
    else:
        if tail_estimators["hill"]!=0:
            final_estimator = tail_estimators["hill"] - 1.
            print("This distribution is heavy-tailed with tail index equal to {} (Adjusted Hill estimated tail index)".format(final_estimator))
        elif tail_estimators["moments"]!=0:
            final_estimator = tail_estimators["moments"] - 1.
            print("This distribution is heavy-tailed with tail index equal to {} (Moments estimated tail index)".format(
                    final_estimator))
        else:
            final_estimator = tail_estimators["kernel"] - 1.
            print(
                "This distribution is heavy-tailed with tail index equal to {} (Kernel-type estimated tail index)".format(
                    final_estimator))
            
    plt.close()
    plt.interactive(True)
    
    return final_estimator
