from .datahandler import UnfoldingDataHandlerLite
from .graphics import print_GRAVEL, print_SAND_II, print_SPUNIT

import numpy as np
from numpy import log as ln
from numpy import exp

class SAND_II(UnfoldingDataHandlerLite):
    minimum_desired_chi2 = 1E-25
    available_methods = [None]
    _essential_attr_list = ['N_meas', 'Rm', 'apriori', 'covar_N', 'desired_chi2'] # actually, covar_N is still needed in order to calculate the chi2 properly.
    def __init__(self, *args, desired_chi2=minimum_desired_chi2, max_iter=100000, **kwargs):
        super().__init__(*args, desired_chi2=desired_chi2, **kwargs)
        self.max_iter = max_iter
        self.num_steps = 0
        self._covar_phi_inv = self.Rm.T @ self.covar_N_inv @ self.Rm
        # self.rho = self.sigma_N / self.N_meas
        self.rho = np.ones(self.m)
        self._append_phi(self.apriori)

    def take_step(self):
        """
        See equation 11 in https://doi.org/10.1016/S0168-9002(01)01438-3 for more details.
        """
        phi = self.phi[-1]
        N_whb = self.Rm @ phi # $N_{would-have-been}$
        weight = ((self.Rm * phi).T / N_whb).T
        lambda_k = 1/(np.ones(self.m) @ (weight.T/self.rho**2).T) # this iteration of the lambda vector. Lenght = n
        d_ln_phi = lambda_k * ((ln(self.N_meas) - ln(N_whb)) @ (weight.T / self.rho**2).T) # 
        new_phi = phi * exp(d_ln_phi)

        self._append_phi(new_phi)
        self.num_steps += 1

    def run(self, method=available_methods[0]):
        while self.num_steps <= self.max_iter:
            self.take_step()
            if self.solution.chi2<=self.desired_chi2:
                break

    def _print_method(self):
        print_SAND_II()

class GRAVEL(SAND_II):
    minimum_desired_chi2 = 1E-25
    available_methods = [None]
    _essential_attr_list = ['N_meas', 'Rm', 'apriori', 'covar_N', 'desired_chi2']
    def __init__(self, *args, desired_chi2=minimum_desired_chi2, max_iter=100000, **kwargs):
        super().__init__(*args, desired_chi2=desired_chi2, max_iter=max_iter, **kwargs)
        self.rho = self.sigma_N / self.N_meas

    def _print_method(self):
        print_GRAVEL()

class SPUNIT(SAND_II):
    _essential_attr_list = ['N_meas', 'Rm', 'apriori', 'covar_N', 'desired_chi2']
    minimum_desired_chi2 = 1E-25
    available_methods = [None]
    def __init__(self, *args, desired_chi2=minimum_desired_chi2, max_iter=100000, **kwargs):
        super().__init__(*args, desired_chi2=desired_chi2, max_iter=max_iter, **kwargs)
        self.rho = self.sigma_N / self.N_meas
        
    def _print_method(self):
        print_SPUNIT()