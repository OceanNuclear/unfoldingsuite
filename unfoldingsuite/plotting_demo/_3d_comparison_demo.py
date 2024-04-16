import sys, os, itertools
sys.path.append("..")
from datahandler import UnfoldingDataHandlerLite, UnfoldingDataLoader
from maximumentropy import MAXED, IMAXED, AMAXED
from linearleastsquare import PseudoInverse
from interface import UMG33Interface
from metrics import Metrics
import numpy as np
from numpy import array as ary
from numpy import sqrt

from drawing_3d import plot_chi2_line, get_singular_dir, prepare_graph
import matplotlib.pyplot as plt

def add_noise(mean_value_array, size=1):
    """
    Create poisson noise around the input (1D) array, and output as 2D array, with number of samples = size
    """
    return np.random.normal(mean_value_array, sqrt(mean_value_array), size=[size, len(mean_value_array)])

num_samples = 40
# Let there be a single truth.
truth = ary([2000, 2000, 2000])
# Use a single response matrix (assume there this is entirely accurate representation)
R = ary([[0,0,1],
        [1,10,0]])
N_true = R @ truth # hidden from program

# A number of experimenters analyse the data, each one gives a different apriori.
apriori_list = add_noise(truth, num_samples)
# Assuming we measure perfectly each time, then we can just plug in N_meas = N_true every time.

def unfold(Program, loader, xr_arg=None):
    """
    condense 3 lines of 
    """
    instance = Program.copy_from(loader)
    if xr_arg:
        instance.run(instance.available_methods[0], xr_arg)
    else:
        instance.run(instance.available_methods[0])
    return instance.solution.phi, instance.solution.chi2

def unfold_3D_view(alg_list, chi2_test_list, extra_args):
    # let's create 100 3 bin spectra
    for Algorithm, xr_arg in itertools.zip_longest(alg_list, extra_args):
        covar_N = np.diag([800, 800])
        covar_N = np.diag(N_true)
        for desired_chi2 in chi2_test_list:
            solutions = []
            for ap_chosen in apriori_list:
                loader = UnfoldingDataLoader(N_meas=N_true, Rm=R, apriori=ap_chosen, covar_N=covar_N, verbosity=0)
                loader.desired_chi2 = desired_chi2
                try:
                    solutions.append(unfold(Algorithm, loader, xr_arg)[0])
                except (IndexError, RuntimeError):
                    solutions.append(ap_chosen)
            fig, ax = prepare_graph(apriori_list, truth)
            ax.scatter(*ary(solutions).T, "solutions")
            for ap, sol in zip(apriori_list, solutions):
                ax.plot(*ary([ap, sol]).T, color="black")
            plot_chi2_line(ax, R, truth, np.arange(0.1, 0.9001, 0.1), chi2_mark=[1.0], covar_N=covar_N)
            title_text = "Unfolding 3-bin spectra with 2 reactions by\n"
            title_text+= repr(Algorithm)[8:-2].split(".")[-1]
            if xr_arg:
                title_text+= " - "+os.path.basename(xr_arg)
            title_text+= "\nran to "+r"$\chi^2$="+str(desired_chi2)
            plt.title(title_text); plt.legend(); plt.show()

def vary_chi2(alg_list, desired_chi2_list):
    for Algorithm in alg_list:
        met = Metrics()
        d_kl = []
        for chi2 in desired_chi2_list:
            for N_meas in N_meas_list:
                loader = UnfoldingDataLoader(N_meas=N_meas, Rm=R, apriori=np.random.choose(apriori_list),
                                            covar_N=np.diag(N_meas))
                met.D_KL(unfold(Program, loader)[0], apriori)


if __name__=="__main__":
    alg_list = UMG33Interface, UMG33Interface, UMG33Interface, IMAXED, AMAXED, PseudoInverse, MAXED
    # alg_list = IMAXED, AMAXED, PseudoInverse, UMG33Interface, UMG33Interface, MAXED
    if sys.argv[1]=="3D":
        extra_args = ("UMG3.3_source/check_behaviour/maxed", "UMG3.3_source/check_behaviour/maxed_modpot", "UMG3.3_source/check_behaviour/maxed_mc")
        chi2_test_list = [2.0, 1.0, 0.5]
        unfold_3D_view(alg_list, chi2_test_list, extra_args)

    elif sys.argv[1]=="chi2":
        pass