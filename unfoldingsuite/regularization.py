from .datahandler import SolutionExtractor, UnfoldingDataHandlerLite
from .graphics import print_REGULARIZER

import numpy as np
from numpy import sqrt, exp
from numpy import array as ary
from numpy import log as ln
from numpy.linalg import inv, norm
from scipy.optimize import root_scalar
import time, copy

nan2num = lambda x: np.nan_to_num(x, nan=0.0, posinf=np.inf, neginf=-np.inf)

class SolutionExtractor_regularization(SolutionExtractor):
    @property
    def loss(self):
        return self.outerclass.loss_val[-1]

class Regularizer(UnfoldingDataHandlerLite):
    available_methods = ['fix_chi2', 'fix_tau']
    minimum_desired_chi2 = 1E-25
    def __init__(self, *args, desired_chi2=minimum_desired_chi2, tau=1, max_iter=2000, **kwargs):
        """
        - required inputs:
        N_meas, Rm, apriori
        sigma_N/covar_N
        desired_chi2
        - optional inputs:
        max_iter, tau
        """
        super().__init__(*args, desired_chi2=desired_chi2, **kwargs)
        self.max_iter = max_iter
        self.tau = tau
        self.fDEF = ary(self.apriori)/sum(self.apriori)
        #create extra lists to record progress
        self.tau_records = []
        self.D_KL = []

        self.solution = SolutionExtractor_regularization(self)
        # take the zero-th step
        self._append_phi(self.apriori)
        self.step_size.pop()

    @property # tau
    def tau(self):
        return self._tau
    @tau.setter
    def tau(self, candidate_value):
        if candidate_value is None:
            self._tau = None
        else:
            assert candidate_value>0, "Tikhonov regularization parameter tau must be a positive number."
            self._tau = candidate_value

    @property # loss_value
    def loss_val(self):
        length = min([len(self.chi2_val), len(self.D_KL)])
        return ary(self.chi2_val[-length:]) + self.tau * ary(self.D_KL[-length:])

    def _append_phi(self, phi, step_size=1):
        self.D_KL.append(self._get_D_KL(phi))
        self.step_size.append(step_size)
        super()._append_phi(phi)
        self.tau_records.append(self.tau)

    def _pop_phi(self):
        """
        Undo the self._append_phi command. Mainly used for backtrack.
        """
        self.D_KL.pop()
        self.step_size.pop()
        super()._pop_phi()
        self.tau_records.pop()

    def _get_D_KL(self, phi):
        """
        Get the D_KL (Kullback-Leibler divergence) value by comparing a phi with the a priori
        """
        f = ary(phi)/sum(phi)
        return np.clip(self.fDEF @ (ln(self.fDEF)-ln(f)), 0, None)

    def Jacobian(self, phi):
        """
        Get the gradient of the loss function
        """
        jac = 2* (self.Rm @ phi - self.N_meas) @ self.covar_N_inv @ self.Rm - self.tau * self.fDEF / phi
        jac += self.tau * 1/phi.sum() * np.ones(self.n) #accounts for the variation in the D_KL function as the phi flux is changed.
        return jac

    def Hessian(self, phi):
        """
        Get the curvature of the loss function (second derivative, self.n * self.n matrix.)
        In underdetermined cases, the determinant of the Hessian matrix will be very sensitive to self.tau.
        If self.tau is close to zero it may be nearly singular and therefore cannot be inverted.
        """
        hess = 2*self.covar_phi_inv + self.tau * np.diag(nan2num(self.fDEF/phi**2)) 
        hess -= self.tau/phi.sum()**2 * np.ones([self.n, self.n]) #accounts for the variation in the D_KL function as we reduce the size.
        return hess

    def take_step(self, wolfe_method=None, **kwargs):
        """
        Take a step foward
        The 'wolfe_method' in the call signature refers to the method argument supplied to the scipy.optimize.root_scalar function inside the self.wolfe_condition.
        For full list of arguments that you can parse to method, see
        """
        d_phi = self.newton(self.phi[-1])
        step_size = self.wolfe_condition(wolfe_method, self.phi[-1], d_phi, **kwargs)
        self._append_phi(self.phi[-1] + d_phi * step_size, step_size)

    def newton(self, phi):
        """
        Calculate the step required to go from the current phi to the phi that gives the minimum loss function
        using the second order approximation of the current loss function landscape.
        """
        jac = self.Jacobian(phi)
        hess = self.Hessian(phi)
        return -inv(hess) @ jac

    def wolfe_condition(self, wolfe_method, phi0, d_phi, upper=2, **kwargs):
        """
        wolfe_method is used as scipy.optimize.root_scalar(method=wolfe_method)
        allowed strings for wolfe_methods are listed in scipy.optimize.show_options('root_scalar')
        'None' can be used as well.
        upper = upper end of the search window within which the true optimum should lie.
        """
        def first_deriv(beta):
            v = d_phi/(phi0 + beta*d_phi)
            v_sum_ratio = d_phi.sum()/(phi0.sum() + beta*d_phi.sum())

            chi2_part = 2 * (self.Rm @ phi0 - self.N_meas) @ self.covar_N_inv @ (self.Rm @ d_phi) + 2 * beta * (d_phi @ self.covar_phi_inv @ d_phi) 
            reg_part = self.tau * (- v @ self.fDEF + v_sum_ratio)
            return chi2_part + reg_part

        def second_deriv(beta):
            v = d_phi/(phi0 + beta*d_phi)
            v_sum_ratio = d_phi.sum()/(phi0.sum() + beta*d_phi.sum())

            chi2_part = 2* (d_phi @ self.covar_phi_inv @ d_phi)
            reg_part = self.tau * ((v**2) @ self.fDEF - v_sum_ratio**2)
            return chi2_part + reg_part

        # a small subroutine to find the true optimum
        max_step_size_before_going_negative = -1/np.clip(
                                    (d_phi/phi0).min(),
                                    -np.inf,
                                    -1/upper) # largest step the algorithm can take before reaching a negative value
        # upper end of the search window within which the root can be found
        max_step_size_before_going_negative -= np.finfo(max_step_size_before_going_negative).eps
        upper = min([max_step_size_before_going_negative, upper])
        if np.sign(first_deriv(0))==1: # the case where d_phi pointed in the wrong direction, i.e. uphill.
            if self.verbosity>=2:
                print("The d_phi_long is pointing in the wrong direction/ something went wrong. Searching in the step size = [-{}, 0] range instead...".format(upper))
                # print("The d_phi_long is pointing in the wrong direction/ something went wrong. Taking a tentative forward step instead...".format(upper))
            return -self.wolfe_condition(wolfe_method, phi0, -d_phi, upper, **kwargs)
            # return np.finfo(upper).resolution
        elif np.sign(first_deriv(upper)) == np.sign(first_deriv(0)) and np.isfinite(first_deriv(upper)):
            if self.verbosity>=2:
                print("The optimum lies beyond the range [0,{}]. Using step size = {}.".format(upper, upper))
            return upper # return unity step size
        elif np.isnan([first_deriv(0), first_deriv(upper)]).any(): # the optimum point lies between step_size=0 and step_size=2
            raise RuntimeError("Invalid steepness value, possibly a singularity due to having zero/negative flux in one bin, encoutnered at step {}!".format(len(self.step_size)+1))
        else: # the optimum point lies at 0<step_size<upper.
            upper -= np.finfo(upper).eps
            if self.verbosity>=2:
                print("The actual optimum lies somewhere between the suggested optimum and the current position. Optimizing...")
            assert np.isfinite(first_deriv(0)), "At step {}, starting point has slope ={} which cannot be used for root search due to having infinities!".format(len(self.step_size)+1, first_deriv(0))
            if not np.isfinite(first_deriv(upper)):
                upper /= 2 # reduce the search window 
                return self.wolfe_condition(wolfe_method, phi0, -d_phi, upper, **kwargs)
            else:
                if self.verbosity>=2:
                    print("Performing 1D optimization within step size bracket = [0, {}]".format(upper))
                result = root_scalar(first_deriv, x0=0, fprime=second_deriv, bracket=[0,upper], method=wolfe_method, **kwargs)
                if self.verbosity>=3:
                    print("Optimization result =\n", result)
                return result.root

    def run(self, method=available_methods[0], tau_search_method=None, num_climb_tol:int = 1, **kwargs):
        """
        available_methods consists of 'fix_tau' and 'fix_chi2'.
            fix_tau is to fix tau and run til convergence.
            fix_chi2 is to run until the final chi2 = desired_chi2.
        tau_search_method specify the root_scalar algorithm used when searching for the correct value of tau.
        kwargs can include wolfe_method and other arguments which iwll be fed root_scalar for the wolfe_method.
        """
        if self.verbosity>=1:
            starttime = time.time()
        if method=='fix_tau':
            if self.verbosity>=2:
                print("Finding the solution for tau=", self.tau)

            self.take_step(**kwargs)
            for num_iter in range(1, self.max_iter):
                self.take_step(**kwargs)
                if np.sign(np.diff(self.loss_val[-2:]))[0] != -1:
                    num_climb_tol -= 1
                    if num_climb_tol < 0:
                        break
            else:
                # if the termination condition is not reached:
                if self.verbosity>=1:
                    raise Exception("Not converged after {} iterations.".format(self.max_iter))

            while np.sign(np.diff(self.loss_val[-2:]))[0] > 0: # backtrack until we reach the nearest minimum loss value iteration
                self._pop_phi()
                num_iter -= 1

            if self.verbosity>=1:
                print("Time taken = {}, number of steps taken = {}".format(time.time()-starttime, num_iter+1))
            return self.phi[-1]

        elif method=='fix_chi2': 
            self.verbosity -= 1 # reduce verbosity to minimize the number of printing
            def extract_chi2(expo_tau):
                self.tau = expo_tau
                self.run('fix_tau', num_climb_tol=num_climb_tol, **kwargs)
                return self.chi2_val[-1]

            extract_chi2(self.tau)
            search_direction = np.sign(self.chi2_val[-1]-self.desired_chi2) # take advantage of the monotonicity of final chi^2 achievable wrt. tau

            while np.sign(self.chi2_val[-1] - self.desired_chi2) == search_direction:
                prev_chi2 = self.chi2_val[-1].copy()
                extract_chi2(self.tau * 10**(-search_direction))

            if self.verbosity>=1-1:
                print("Search bracket values of tau are [{},{}]".format(self.tau, self.tau*10**(search_direction)))
                print("which gives chi^2 values of [{}, {}]".format(self.chi2_val[-1], prev_chi2))

            reach_required_chi2 = lambda chi2: np.log10(extract_chi2(chi2)) - np.log10(self.desired_chi2)

            result = root_scalar(reach_required_chi2,
                                x0=self.tau,
                                bracket=[self.tau, (self.tau)* 10**(search_direction)],
                                method=tau_search_method,)
            self.verbosity += 1
            if self.verbosity>=2:
                print("Result obtained after searching for the appropriate value of chi^2 to obtain the required tau=\n", result)
            extract_chi2(result.root) # run it one final time just to be sure that the last phi will be geneated from the correct value of tau.

            if self.verbosity>=1:
                print("Time taken = {}, total number of steps taken = {}".format(time.time()-starttime, len(self.step_size)))
            return result.root

        else:
            print("method not recognized. Please check self.available_methods=\n", available_methods, "\nFor more details, see scipy.optimize.show_options('root_scalar')")

    def _print_method(self):
        print_REGULARIZER()

"""
Pending: remove the 'debug_escape' ?
"""