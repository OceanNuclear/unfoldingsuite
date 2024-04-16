import math
import numpy as np
from numpy import array as ary
from numpy.linalg import inv
from numpy import log as ln
from numpy import sqrt

def cast_into_numpy_array(method):
    """make sure that the inputs are given in form of a numpy array"""
    def modified_method(self, *args, **kwargs):
        return method(self, *[ary(arg) for arg in args], **{k:ary(v) for (k,v) in kwargs.items()})
    return modified_method

class FractionalDeviation():
    @cast_into_numpy_array
    def __init__(self, vector_dev):
        """
        convert the vector of fractional deviations into 
        """
        self.deviations = vector_dev
        self.mean_abs = abs(vector_dev).mean()
        self.mean_abs_log = abs(ln(vector_dev+1)).mean()
        self.root_mean_squared = sqrt(((vector_dev)**2).mean())
        self.root_mean_squared_log = sqrt(((ln(vector_dev+1))**2).mean())

class Metrics():
    def __init__(self, clip_mask=None):
        """
        use clip_maks to ignore a part of the spectrum that we don't care about.
        """
        self.clip_mask=clip_mask

    def clip_spectrum(self, spectrum):
        if self.clip_mask is not None:
            return self.clip_mask * spectrum
        else:
            return spectrum

    @cast_into_numpy_array
    def D_KL(self, test, truth):
        test, truth = self.clip_spectrum(test), self.clip_spectrum(truth)
        """
        D_KL = Kullback-Leibler Divergence
        measures the difference between
        After a careful review of the definition of D_KL and cross-entropy {Ch.2 of Kenneth Burnhm ISBN:9780387953649}, 
        I have concluded that in case of unfolding, the apriori should be used in place of the truth, not the test, distribution.
        Because the a priori is the only piece of known information that we can rely on.
        """
        # normalize the two distributions
        truth = truth/sum(truth)
        test = test/sum(test)
        return np.nan_to_num(truth * (ln(truth)-ln(test))).sum()

    @cast_into_numpy_array
    def maxed_cross_entropy(self, test, truth):
        test, truth = self.clip_spectrum(test), self.clip_spectrum(truth)
        """
        {title = "MAXED, a computer code for maximum entropy deconvolution of multisphere
                neutron spectrometer data",
                author = "M. Reginatto and P. Goldhagen",
                journal = "Health Phys.",
                volume = "77",
                number = "5",
                pages = "579-83",
                year = "1999"}
        MAXED defines cross-entropy differently . They use the following equation:
        """
        #normalize only one spectrum.
        truth_yield = sum(truth)
        test_yield = sum(test)
        test_dist = test/truth_yield
        truth_dist = truth/truth_yield
        # return test_dist * (ln(truth_dist)-ln(test_dist)).sum() - sum(truth_dist) + sum(test_dist) # where sum(truth) is, by definition
        return np.nan_to_num(test_dist * (ln(truth_dist)-ln(test_dist))).sum() - sum(test_dist) -1 # = S in Appendix A, where S is to be maximized.

    @cast_into_numpy_array
    def chi2_from_N(self, N_test, N_meas, std_or_variance_matrix):
        """compare the difference between two {N} vectors"""
        if std_or_variance_matrix.ndim==1:
            covar_N = np.diag(std_or_variance_matrix**2)
        elif std_or_variance_matrix.ndim==2:
            covar_N = std_or_variance_matrix
        else:
            raise ValueError('The input to the std_or_variance_matrix argument must be either the standard deviation vector (sigmas, assuming no covariance) or the covariance matrix.')
        return (N_meas-N_test) @ inv(covar_N) @ (N_meas-N_test)

    @cast_into_numpy_array
    def chi2_from_spectrum_and_N(self, Rm, phi_test, N_meas, std_or_variance_matrix):
        """calculate chi^2 from a test spectrum and the measured N"""
        return self.chi2_from_N(N_meas, Rm @ self.clip_spectrum(phi_test), std_or_variance_matrix)

    @cast_into_numpy_array
    def chi2_from_spectrum(self, Rm, phi_test, phi_true, std_or_variance_matrix):
        phi_test, phi_true = self.clip_spectrum(phi_test), self.clip_spectrum(phi_true)
        """calculate chi^2 from a test spectrum and the true spectrum"""
        return self.chi2_from_N(Rm @ self.clip_spectrum(phi_test), Rm @ self.clip_spectrum(phi_true), std_or_variance_matrix)

    @cast_into_numpy_array
    def fractional_deviation(self, test, truth):
        test, truth = self.clip_spectrum(test), self.clip_spectrum(truth)
        return FractionalDeviation(self.clip_spectrum(test)/self.clip_spectrum(truth) - 1)

    @cast_into_numpy_array
    def split_energy_range(self, old_bin_structure, new_bin_structure, PUL=True):
        """Create a list of Metrics objects, such that the metrics can be evaluated on individual regions of the spectrum."""
        assert new_bin_structure.ndim==1, "Expect group structure to be given in linear form."
        metrics_list = []
        for (lower_E, upper_E) in zip(new_bin_structure[:-1], new_bin_structure[1:]):
            if PUL:
                leth_space = np.diff(ln(old_bin_structure))
                dest_space = np.diff(np.clip(ln(old_bin_structure), ln(lower), ln(upper)))
                mask = dest_space/leth_space
            else:
                E_space = np.diff(old_bin_structure)
                dest_space = np.diff(np.clip(old_bin_structure, lower, upper))
                mask = E_space/leth_space
            metrics_list.append(Metrics(mask))
        return metrics_list # each of the Metrics object measures the deviations etc. in a specific range of energy of the spectrum.

    @cast_into_numpy_array
    def yield_deviation(self, test, truth):
        test, truth = self.clip_spectrum(test), self.clip_spectrum(truth)
        """
        calculate how much does the test spectrum deviate from the truth spectrum
        """
        return np.nan_to_num(test.sum()/truth.sum() -1)

    def yield_deviation(self, test, truth):
        return