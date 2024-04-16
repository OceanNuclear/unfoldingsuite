import matplotlib.pyplot as plt
import pandas as pd
import numpy as np; from numpy import log as ln
from numpy.linalg import inv, det
from numpy import diag, sqrt
import time

from correctedinversemethod import BareRegularizer
from types import MethodType
from scipy.optimize import fsolve
import seaborn as sns
#The following program is 
NUM_SAMPLE = 20
TAU=1E-9
num_tol = np.finfo(TAU).max #largest number that the computer can store safely
apriori = pd.read_csv('a_priori.csv', header=None)
answer  = pd.read_csv('answer_spectrum.csv',header=None).values.reshape([-1])
#load in the group structure
# gs      = 
#load in the response matrix
R       = pd.read_csv('R.csv', header=None, index_col=0)
#reaction rates
rr      = pd.read_csv('reaction_rates.csv', header=None)
sigma   = sqrt(rr.values.reshape([-1]))
def cross_over_search( func, start, stop, depth=0):
    range_to_search = np.linspace(start, stop)
    arg_vec = np.argwhere(np.diff([ func(i) for i in range_to_search ]))
    assert len(arg_vec)==1, f"expected one and only one cross-over point in this range, instead we have {len(arg_vec)}."
    crossing = arg_vec[0][0]
    if depth<10:
        return cross_over_search(func, range_to_search[crossing], range_to_search[crossing+1], depth+1)
    else:
        xmin = range_to_search[crossing]
        xmax = range_to_search[crossing+1]
        if func(xmin): #Stay on the small side of the small side evaluates to True
            return xmin
        elif func(xmax): #Stay on the large side of the large side evaluates to True
            return xmax


class SmartRegularizer(BareRegularizer):
    def __init__(self, *args):
        super().__init__(*args)

    def get_H(self, t, f):
        _Y = self.best_fit_n_yield(f)
        hessian_matrix = ( _Y* self.R.T ) @ self.S_N @ (_Y * self.R) + t * np.diag(self.fDEF/(f**2))
        return hessian_matrix

    def is_matrix_non_singular(self, t,f):
        yes_no = np.isfinite(np.log2( det(self.get_H(t,f)) )) # check if that tau is finite
        return yes_no

    def explore_current_log2_tau(self, log2t):
        return self.is_matrix_non_singular(2**log2t, self.f[-1])

    def get_current_tau_range(self): #this function has to specifically avoid using t
        np.seterr('ignore') #ignore all warning; so this is potentially dangerous
        theoretical_low, theoretical_high = -25, 2  #we're quite sure that it'll be in this range.
        log2t_range = np.linspace(theoretical_low, theoretical_high)
        valid = [ self.explore_current_log2_tau(log2t) for log2t in log2t_range ]
        mid_range = np.mean(log2t_range[valid])
        lowerbound = cross_over_search( self.explore_current_log2_tau, theoretical_low, mid_range )
        upperbound = cross_over_search( self.explore_current_log2_tau, mid_range, theoretical_high )
        np.seterr('print')
        return 2**lowerbound, 2**upperbound

    def copy(self):
        return SmartRegularizer(self.N, self.S_N, self.R, self.fDEF)

if __name__=="__main__":
    starttime = time.time()
    reg = SmartRegularizer(rr.values.reshape([-1]), np.diag(1/sigma**2), R.values, apriori.values.reshape([-1]))
    
    chi2_val, reg_val, iterations = [], [], []
    f = []
    tau_low, tau_high = [], []
    easy_logspace = lambda start, end, num: np.logspace(*np.log10([start, end]), num)
    trange = easy_logspace(*reg.get_current_tau_range(), NUM_SAMPLE)
    # trange = reg.get_current_tau_range()
    # def increased_logspace(start, end, num):
    #     log10_start, log10_end = np.log10([start, end])
    #     return np.logspace(lower_start, log10_end, num)
    reso = np.finfo(reg.tau).resolution
    for t in trange:
        reg = reg.copy()
        reg.tau = t
        while True:
            reg.take_step()
            if all(reg.steepness[-2:] < 5* reso):
                break #break when the last two iterations has a smaller than the possible resolution of the program
        chi2_val.append(reg.chi2_val[-1])
        reg_val.append(reg.reg_val[-1])
        iterations.append(len(reg.f))
        f.append(reg.f[-1])
        print(f"time passed = {time.time() - starttime} s")
        print(t, (min_max := reg.get_current_tau_range()) )
        tau_low.append(min_max[0])
        tau_high.append(min_max[1])
    # plt.loglog(trange, tau_low, label=r"lower bound of $\tau$")
    # plt.loglog(trange, tau_high, label=r"upper bound of $\tau$")
    plt.fill_between(trange, tau_low, tau_high, alpha=0.5, color='C1')
    plt.yscale('log')
    plt.xscale('log')
    plt.title("variation of the upper and lower bound of tau\nfor the solutions reached by different tau's,\nwithin which the LHS_matrix is nonsingular")
    plt.legend()
    plt.show()
        # Thorough investigation shows that both the upper bound and lower bound only drops by a maximum of a factor of 2 after the changing the 

        # Took me 35 minutes to get this done :) 