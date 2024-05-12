from .datahandler import UnfoldingDataHandlerLite
from .graphics import print_GRAVEL, print_SAND_II, print_SPUNIT

import numpy as np
from numpy import log as ln
from numpy import exp

class SAND_IIBase(UnfoldingDataHandlerLite):
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
        N_whb = self.Rm @ phi
        weight = self.get_weight(phi, N_whb)
        lambda_k = 1/(np.ones(self.m) @ (weight.T/self.rho**2).T) # this iteration of the lambda vector. Lenght = n
        d_ln_phi = lambda_k * ((ln(self.N_meas) - ln(N_whb)) @ (weight.T / self.rho**2).T) # 
        new_phi = phi * exp(d_ln_phi)

        self._append_phi(new_phi)
        self.num_steps += 1

    def get_weight(self, phi, N_whb):
        weight = ((self.Rm * phi).T / N_whb).T
        return weight

    def run(self, method=available_methods[0]):
        while self.num_steps <= self.max_iter:
            self.take_step()
            if self.solution.chi2<=self.desired_chi2:
                break

    def _print_method(self):
        print_SAND_II()

class SAND_II(SAND_IIBase):
    minimum_desired_chi2 = 1E-25
    available_methods = [None, "smoothing"]
    def __init__(self, *args, desired_chi2=minimum_desired_chi2, max_iter=100000, **kwargs):
        super().__init__(*args, desired_chi2=desired_chi2, **kwargs)
    
    def get_smoothing_factors(self, Ns):
        # unfortunately, the maths was not formulated in a way that supports vector operation easily.
        # Instead we resort to for-loop with three if-conditions.
        m = self.m
        divisior_vector = np.ones(m)
        for j in range(2, m):# 1<j<m
            j_ = j-1
            if 1 < j <= ((Ns-1)/2):
                divisior_vector[j_] = 2*j - 1
            elif ((Ns-1)/2) < j <= (m - (Ns-1)/2):
                divisior_vector[j_] = Ns
            elif (m - (Ns-1)/2) < j < m:
                divisior_vector[j_] = 2*m - 2*j + 1
            else:
                raise ValueError("ProgrammerError")
        return divisior_vector

    def get_smooth_weight(self, phi, N_whb):
        weights_T = (N_whb / self._smoothing_divisor_factor) * (self.Rm*phi).T
        weights_T[0] = (self.Rm[:,:3] @ (phi[:3] * [5,2,-1])) /6 / N_whb
        weights_T[-1] = (self.Rm[:,-3:] @ (phi[-3:] * [-1,2,5])) /6 /N_whb
        return weights_T.T

    def take_smooth_step(self, Ns):
        """
        See equation 11 in https://doi.org/10.1016/S0168-9002(01)01438-3 for more details.
        """
        phi = self.phi[-1]
        self._smoothing_divisor_factor = self.get_smoothing_factors(Ns)
        N_whb = self.Rm @ phi
        weight = self.get_smooth_weight(phi, N_whb)
        lambda_k = 1/(np.ones(self.m) @ (weight.T/self.rho**2).T) # this iteration of the lambda vector. Lenght = n
        d_ln_phi = lambda_k * ((ln(self.N_meas) - ln(N_whb)) @ (weight.T / self.rho**2).T) # 
        new_phi = phi * exp(d_ln_phi)

        self._append_phi(new_phi)
        self.num_steps += 1

    def run(self, method=available_methods[0], Ns=0):
        if method is None:
            return super().run()
        elif method=="smoothing":
            assert Ns>0, "The degree of smoothing must be an integer of at least 1."
            while self.num_steps <= self.max_iter:
                self.take_smooth_step(Ns)
                if self.solution.chi2<=self.desired_chi2:
                    break

class GRAVEL(SAND_IIBase):
    minimum_desired_chi2 = 1E-25
    available_methods = [None]
    _essential_attr_list = ['N_meas', 'Rm', 'apriori', 'covar_N', 'desired_chi2']
    def __init__(self, *args, desired_chi2=minimum_desired_chi2, max_iter=100000, **kwargs):
        super().__init__(*args, desired_chi2=desired_chi2, max_iter=max_iter, **kwargs)
        self.rho = self.sigma_N / self.N_meas

    def _print_method(self):
        print_GRAVEL()

class SPUNIT(SAND_IIBase):
    _essential_attr_list = ['N_meas', 'Rm', 'apriori', 'covar_N', 'desired_chi2']
    minimum_desired_chi2 = 1E-25
    available_methods = [None]
    def __init__(self, *args, desired_chi2=minimum_desired_chi2, max_iter=100000, **kwargs):
        super().__init__(*args, desired_chi2=desired_chi2, max_iter=max_iter, **kwargs)
        self.rho = self.sigma_N / self.N_meas
        
    def _print_method(self):
        print_SPUNIT()