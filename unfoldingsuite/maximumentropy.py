from .datahandler import UnfoldingDataHandlerLite, SolutionExtractor
from .graphics import print_MAXED, print_IMAXED, print_MAXED_baseclass, print_AMAXED

from numpy import array as ary
from numpy import log as ln
from numpy import sqrt, exp, mod, float32, float64
import numpy as np
import time
import random
from numpy.linalg import inv, norm, pinv
from scipy.stats import linregress
from scipy.optimize import root_scalar, minimize

class SolutionExtractor_MAXED(SolutionExtractor):
    @property
    def lambda_vector(self):
        return self.outerclass.lambda_vector[-1]

    @property
    def entropy(self):
        return self.outerclass.entropy[-1]

    @property
    def covar_phi_inv(self):
        covar_phi_inv = self.outerclass.get_covar_phi_inv(self.lambda_vector)

        if np.linalg.matrix_rank(covar_phi_inv)==np.shape(covar_phi_inv)[0]:
            return inv(covar_phi_inv)
        else:
            return pinv(covar_phi_inv)
class SolutionExtractor_AMAXED(SolutionExtractor):
    """
    custom designed for AMXAXED, returns the solution spectrum phi without including the lagrangian multiplier mu at the end.
    """
    @property
    def phi(self):
        return self.outerclass.phi[-1][:-1]

    @property
    def covar_phi(self):
        """
        Get the covariance matrix of phi,
        """
        return inv(self.outerclass.Hessian(self.outerclass.phi[-1])[:-1,:-1])

class AMAXED(UnfoldingDataHandlerLite):
    available_methods = [None, 'brenth', 'brentq', 'ridder', 'toms748', 'newton'] # methods to control the step-size
    def __init__(self, *args, desired_chi2=0, max_iter=2000, **kwargs):
        super().__init__(*args, desired_chi2=desired_chi2, **kwargs)
        """
        Note that the self.phi variable stores a list of vector, each with length = self.n+1,
        where the first n elements is the actual flux at that iteration; while the last element is the lagrangian multiplier mu.
        """
        self.solution = SolutionExtractor_AMAXED(self) # change the extractor to add more functionality
        self.max_iter = max_iter # add max_iter argument

        # make new lists for recording data
        self.loss_val, self.D_KL = [], []

        # functions/constants which are only used during the rest of the calculations
        self._a = self.N_meas @ self.covar_N_inv @ self.Rm
        self._get_b = lambda phi: phi @ self.covar_phi_inv
        self._c = self.N_meas @ self.covar_N_inv @ self.N_meas

        #calculate the phi_long for the zero-th iteration.
        self.fDEF = ary(self.apriori)/sum(self.apriori)
        best_fit_yield = (self.N_meas @ self.covar_N_inv @ (self.Rm @ self.fDEF)) / (self.fDEF @ self.covar_phi_inv @ self.fDEF)
        first_phi = self.fDEF * best_fit_yield
        self.phi.append(first_phi)
        mu = self._get_mu(first_phi, average_appraoch=False)

        # record the intitial 0th step
        self._append_phi( ary(first_phi.tolist()+[mu,]) )
        self.step_size.pop() # the first 'step' doesn't count; it's just initialization.

    def _append_phi(self, phi_long, step_size=1):
        """
        append phi_long onto phi, and use phi to calculate the rest of the quantities and record them.
        """
        phi = phi_long[:-1]
        self.D_KL.append(self._get_D_KL(phi))
        self.step_size.append(step_size)

        self.phi.append(phi_long)
        self.N.append(self.Rm@phi)
        self._get_chi2(append=True)
        self.loss_val.append(self._get_loss(phi_long))

    def _pop_phi(self):
        self.D_KL.pop()
        self.step_size.pop()
        super()._pop_phi()
        self.loss_val.pop()

    def _get_D_KL(self, phi):
        """
        Get the D_KL (Kullback-Leibler divergence) value by comparing a phi with the a priori
        """
        f = ary(phi)/phi.sum()
        return self.fDEF @ ln(self.fDEF / f)

    def _get_loss(self, phi_long):
        """
        return the Lagrangian value,
        i.e. the scalar quantity that is being extremized in the Lagrangian Multiplier method.
        """
        mu = phi_long[-1]
        phi = phi_long[:-1]
        lagrangian = -self.fDEF@ln(phi) + ln(phi.sum()) + mu*(self._get_chi2(test_phi=phi) - self.desired_chi2)
        return lagrangian

    def _get_mu(self, phi, average_appraoch=False):
        """
        Get the lagrangian multiplier mu.
        use average_appraoch when we're NOT at a stationary point.
        """        
        a = self._a
        b = self._get_b(phi)

        if average_appraoch:
            mu = (self.fDEF/phi - 1/phi.sum()) / (2*b - 2*a)
            mu = np.mean(mu)
        else:
            mu = ((b-2*a) @ self.fDEF + (self._c - self.desired_chi2)/phi.sum()) /2 /((b**2) @ phi - 3*(b*a)@phi + 2 * (a**2) @ phi)
        return mu

    def Jacobian(self, phi_long):
        """
        Find Del * Lagrangian
        """
        mu = phi_long[-1]
        phi = phi_long[:-1]

        a = self._a
        b = self._get_b(phi)

        jac = np.zeros(self.n+1)
        jac[:-1] = 1/phi.sum() - self.fDEF/phi + 2*mu * (b-a)
        jac[-1] = self._get_chi2(test_phi=phi) - self.desired_chi2
        return jac

    def Hessian(self, phi_long):
        """
        get the Del^2 Lagrangian
        """
        mu = phi_long[-1]
        phi = phi_long[:-1]

        a = self._a
        b = self._get_b(phi)

        hess = np.zeros([self.n+1, self.n+1])
        hess[:-1, :-1] = -np.ones([self.n, self.n])/phi.sum()**2 + np.diag(self.fDEF/phi**2) + 2*mu * self.covar_phi_inv
        hess[:-1,-1] = hess[-1,:-1] = 2*(b-a)
        return hess

    def take_step(self, method, **kwargs):
        """
        take a step using the method stated, and add the result to self.phi
        """
        d_phi_long = -inv(self.Hessian(self.phi[-1])) @ self.Jacobian(self.phi[-1])
        step_size = self.wolfe_condition(method, self.phi[-1], d_phi_long)
        new_phi = self.phi[-1] + step_size * d_phi_long
        self._append_phi(new_phi, step_size)

    def newton(self, phi_long):
        """
        Calculate the step required to go from the current phi to the phi that gives the minimum loss function
        using the second order approximation of the current loss function landscape.
        """
        jac = self.Jacobian(phi_long)
        hess = self.Hessian(phi_long)
        return -inv(hess) @ jac

    def wolfe_condition(self, method, phi_long, d_phi_long, upper=2, **kwargs):
        mu = phi_long[-1]
        phi = phi_long[:-1]
        d_mu = d_phi_long[-1]
        d_phi = d_phi_long[:-1]

        a = self._a
        b = self._get_b(phi)

        def first_deriv(beta):
            total_flux = phi + beta*d_phi
            return d_phi.sum()/total_flux.sum() - (self.fDEF/total_flux) @ d_phi + d_mu * (self._get_chi2(test_phi=total_flux)-self.desired_chi2) + (mu + beta*d_mu) * ( 2*(b-a)@d_phi + 2*beta * d_phi@self.covar_phi_inv@d_phi )

        def second_deriv(beta):
            total_flux = phi + beta*d_phi
            return - (d_phi.sum()/total_flux.sum())**2 + ((self.fDEF*d_phi)/total_flux**2) @ d_phi + d_mu*( 2*(b-a)@d_phi + 2*beta * d_phi@self.covar_phi_inv@d_phi ) + (mu + beta*d_mu) * (2 * d_phi@self.covar_phi_inv@d_phi)

        upper_max = -1/np.clip( (d_phi/phi).min(), -np.inf, -1) # upper end of the search window where the root should be found
        upper = min([upper_max, upper])
        # if np.sign(first_deriv(0))==-1: # if the d_phi is miscalculated and we ended up pointing in the uphill direction
        #     if self.verbosity>=2:
        #         print("The d_phi_long is pointing in the wrong direction. Searching in the step size = [0, -{}] range instead...".format(upper))
        #     return -self.wolfe_condition(method, phi_long, -d_phi_long, upper, **kwargs)
        if np.sign(first_deriv(upper)) == np.sign(first_deriv(0)):
            pass # return unity step size
        elif np.isnan([first_deriv(0), first_deriv(upper)]).any(): # the optimum point lies between step_size=0 and step_size=2
            pass # return unity step size
        else: # the optimum point lies at step_size>2.
            if self.verbosity>=2:
                print("The actual optimum lies somewhere between the suggested optimum and the current position. Optimizing...")
            assert np.isfinite(first_deriv(0)), "Starting point has slope ={} which cannot be used for root search due to having infinities!".format(first_deriv(0))
            while not np.isfinite(first_deriv(upper)):
                upper /= 2 # reduce the search window 
            if self.verbosity>=2:
                print("Performing 1D optimization within step size bracket = [0, {}]".format(upper))
            result = root_scalar(first_deriv, x0=0, bracket=[0,upper], method=method, **kwargs)
            if self.verbosity>=3:
                print("Optimization result =\n", result)
            return result.root

        if self.verbosity>=2:
            print("The optimum lies beyond the range [0,{}]. Using step size of unity.".format(upper))
        return 1

    def _print_method(self):
        print_AMAXED()

    def run(self, method=available_methods[0], num_climb_tol:int = 5, **kwargs):
        """
        Run until we reach the minimum loss value.
        """
        if self.verbosity>=1:
            starttime = time.time()

        for num_iter in range(self.max_iter):
            self.take_step(method, **kwargs)
            if np.sign(np.diff(abs(ary(self.chi2_val[-2:]) - self.desired_chi2)))[0] != -1:
                num_climb_tol -= 1
                if num_climb_tol < 0:
                    break
        else:
            # if the termination condition is not reached:
            if self.verbosity>=1:
                raise Exception("Not converged after {} iterations.".format(self.max_iter))

        while np.sign( np.diff(abs(ary(self.chi2_val[-2:]) - self.desired_chi2)))[0] > 0: # backtrack until we reach the nearest minimum loss value iteration
            self._pop_phi()
            num_iter -= 1

        if self.verbosity>=1:
            print("Time taken = {}, number of steps taken = {}".format(time.time()-starttime, num_iter+1))
        return self.phi[-1]

class _MAXED_baseclass(UnfoldingDataHandlerLite):
    def __init__(self, *args, desired_chi2=0, **kwargs):
        super().__init__(*args, desired_chi2 = desired_chi2, **kwargs)
        # change the extractor to add more functionality
        self.solution = SolutionExtractor_MAXED(self)

        # make new lists for recording data
        self.potential = []
        self.entropy = []
        self.lambda_vector = []

        # rescale apriori into an fDEF with best fit yield.
        N_ap = self.Rm @ self.apriori
        # self._apriori_scale_factor = (self.N_meas @ self.covar_N_inv @ N_ap)/(N_ap @ self.covar_N_inv @ N_ap)
        self._apriori_scale_factor = 1
        self.fDEF = self._apriori_scale_factor * ary(self.apriori)

        # the omega constant = self.desired_chi2
        # the value of desired_chi2 must be provided before the loss function evaluators work.
        # By default we will assume desired_chi2 = 0

        # take a random step, so that we
        # start at a random distribution of numbers between -0.01 to 0.01; because we can't start at exactly all zeros.
        self._append_lambda((np.random.random(self.m)*2-1) * 0.01) # record the intitial 0th step
        self.step_size.pop()

    def _append_lambda(self, l_vec, step_size=1):
        self.lambda_vector.append(l_vec.copy())
        self.step_size.append(step_size)
        self.potential.append(self.potential_function(l_vec))
        new_phi = self.lambda2spectrum(l_vec)
        super()._append_phi(new_phi)
        self.entropy.append( self._get_entropy(new_phi) )

    def _pop_lambda(self):
        self.lambda_vector.pop()
        self.step_size.pop()
        self.potential.pop()
        super()._pop_phi()
        self.entropy.pop()

    #conversion between the lambda vector and other scalar quantities/vectors
    def potential_function(self, l_vec):
        l_vec = ary(l_vec)
        exponent = (l_vec) @ self.Rm
        exponent = np.clip(exponent, -180, 174)
        sum1 = self.fDEF @ exp(-exponent) # dot product between fDEF and exp(-lR) # maximize the flux
        sum3 = sqrt(self.desired_chi2 * (l_vec @ self.covar_N @ l_vec)) # ensure the solution.N matches desired chi2
        sum4 = l_vec @ self.N_meas # minimize the lambda vector
        Z = -sum1 -sum3 -sum4
        if self.verbosity>=3:
            print('-sum1, -sum2, -sum3 =', -sum1, -sum3, -sum4)
            print(Z)
            print('----------------------------------')
        return Z

    def lambda2spectrum(self, l_vec):
        exponent = l_vec @ self.Rm
        exponent = np.clip(exponent, -180, 174)
        final_flux = self.fDEF * exp(-exponent)
        return final_flux

    # methods for getting lambda-dependent scalars, vectors, and matrices.
    def Hessian(self, l_vec):
        '''
        method used to get the Hessian matrix at the given lambda vector.
        '''
        b = self.fDEF * exp(-l_vec @ self.Rm)

        invertible_matrix = - self.Rm @ (self.Rm * b).T \
                            + sqrt(self.desired_chi2/(l_vec @ self.covar_N @ l_vec)**3) \
                            * (np.outer(self.covar_N@l_vec, l_vec@self.covar_N) - (l_vec @ self.covar_N @ l_vec)* self.covar_N)
        return invertible_matrix

    def Jacobian(self, l_vec):
        """
        Calculate the Jacobian vector
        """
        b = self.fDEF * exp(-l_vec @ self.Rm)
        return self.Rm @ b - self.N_meas - sqrt(self.desired_chi2/(l_vec@self.covar_N@l_vec)) * self.covar_N @ l_vec

    def get_covar_phi_inv(self, l_vec, alt_def=False):
        """
        Try to get the inverse of the covariance matrix of the output spectrum.
        However I think I've made a mistake on both of them.
        """
        phi = self.lambda2spectrum(l_vec)
        if alt_def:
            Rm_inv = pinv(self.Rm)
            denominator = l_vec @ self.covar_N @ l_vec
            S_phi = Rm_inv @ self.covar_N @ Rm_inv.T

            curvature_of_phi = np.diag(np.ones(self.n) - (Rm_inv@self.N_meas)/phi**2) 
            curvature_of_phi -= sqrt(self.desired_chi2/denominator**3) * (S_phi @ ln(phi/self.fDEF)) * self.fDEF/phi
            curvature_of_phi += sqrt(self.desired_chi2/denominator) * S_phi @ np.diag(self.fDEF*(1-ln(phi/self.fDEF))/phi**2)
        else:
            modifier = 1/phi * self.Rm
            covar_lambda_inv = self.Hessian(l_vec)
            curvature_of_phi = (modifier.T) @ covar_lambda_inv @ modifier
        return - curvature_of_phi

    def _get_entropy(self, phi, exclude_flux_sum=False):
        """
        return the entropy value as defined by Skilling (1989),
        which is a generalization of the cross-entropy formula.
        """
        entropy = - phi @ ln(phi/self.fDEF)
        if not exclude_flux_sum:
            entropy -= sum(self.fDEF) - sum(phi)
        return entropy

    def _print_method(self):
        print_MAXED_baseclass()

class IMAXED(_MAXED_baseclass):
    """
    Uses Newton's method to calculate the direction and magnitude of step delta_lambda required to reach the optimum in the second order approximation of the loss function landscape;
    Reduce the magnitude of the step if necessary to avoid overshoots.
        This is done by applying Wolfe's condition (optimum search on the line // to the direction of the step.)
    """
    available_methods = [None, 'brenth', 'brentq', 'ridder', 'toms748', 'newton'] # methods to control the step-size
    def __init__(self, *args, desired_chi2=0, max_iter=2000, **kwargs):
        super().__init__(*args, desired_chi2=desired_chi2, **kwargs)
        self.max_iter = max_iter

    def run(self, method=available_methods[0], num_climb_tol:int =1, **kwargs):
        """
        method is fed into self.wolfe_condition() as the argument for the root_scalar(method=?)
        num_climb_tol is the number of "uphill" climbs allowed before it terminates.
        This algorithm works fine as ong as num_climb_tol>=1.
        """
        if self.verbosity>=1:
            starttime = time.time()

        for num_iter in range(self.max_iter):
            # monitor the steepness only.
            self.take_step(method, **kwargs)

            if np.sign(np.diff(self.potential[-2:]))[0]!=1: # if it isn't climbing anymore.
                num_climb_tol -= 1 # this is equivalent to loss function increasing.
                if num_climb_tol < 0:  # when we use up our quota of allowed number of climbs
                    break

        else: # if not converged, print the following:
            if self.verbosity>=1:
                print("Not converged after {} iterations. Aborting ...".format(self.max_iter))
            if self.verbosity>=2:
                print("Please re-try with a smaller initial lambda vector; perhaps one with all negatie numbers with small absolute values.")
            if self.verbosity>=3:
                print("This is because underestimation of lambda will lead to underestimation of step size required to change lambda; and vice versa.")

        while np.sign(np.diff(self.potential[-2:]))<0: # whether we've converged or not, do the following:
            # backtrack until we reach the nearest iteration with maximum potential.
            self._pop_lambda()
            num_iter -= 1

        if self.verbosity>=1:
            print("Time taken = {}, number of steps taken = {}".format(time.time()-starttime, num_iter+1))
        
        return

    # methods for the actual optimization

    def take_step(self, method, **kwargs):
        """
        take a step towards the new minimum
        """
        delta_lambda = self.newton(self.lambda_vector[-1])
        step_size = self.wolfe_condition(method, self.lambda_vector[-1], delta_lambda, **kwargs)
        # or use other conditions, e.g. not allowing the deviation between approximation and thing to exceed a certain limit?
        new_lambda = step_size * delta_lambda + self.lambda_vector[-1]
        self._append_lambda(new_lambda, step_size)

    def wolfe_condition(self, method, l0, dl, upper=2, **kwargs):
        """
        Calculates how big of a step to take to reach the minimum in that direction.
        l0 is the starting position vector
        dl is the search direction vector.
        upper = upper end of the search window within which the true minimum will be found.
        """
        b = self.fDEF * exp(-l0 @ self.Rm)
        A = l0 @ self.covar_N @ l0
        B = l0 @ self.covar_N @ dl
        C = dl @ self.covar_N @ dl

        # derivative wrt. beta for loss(l0 + beta*dl)
        first_deriv = lambda beta: (b * (dl @ self.Rm))   @ exp(-beta * dl @ self.Rm) - sqrt(self.desired_chi2/(A + 2*B*beta + C*beta**2))  * (B + C*beta) - self.N_meas @ dl
        second_deriv= lambda beta: -b * (dl @ self.Rm)**2 @ exp(-beta * dl @ self.Rm) - sqrt(self.desired_chi2/(A + 2*B*beta + C*beta**2)**3)*(A*C - B**2)

        # a small subroutine to find the proper area
        if np.sign(first_deriv(0))==-1: # must be going up hill to begin with, since we're trying to maximize.
            return -self.wolfe_condition(method, l0, -dl, upper, **kwargs) # reverse the search direction.
        elif np.sign(first_deriv(upper)) != np.sign(first_deriv(0)): # the optimum point lies between step_size=0 and step_size=2
            if self.verbosity>=2:
                print("The actual optimum lies somewhere between the suggested optimum and the current position. Optimizing...")
            assert np.isfinite(first_deriv(0)), "Starting point cannot be used for root search due to having slope = {} = undefined!".format(first_deriv(0))
            while not np.isfinite(first_deriv(upper)):
                upper /= 2 # reduce the search window until we go back to a finite range
            return root_scalar(first_deriv, x0=0, fprime=second_deriv, bracket=[0,upper], method=method, **kwargs).root
        else: # the optimum point lies at step_size>2.
            return 1

    def newton(self, l_vec):
        r"""
        Calculate the vector $\Delta \Lambda$ that will bring us to the minimum in the second order approaximation.
        """
        delta_lambda = - inv(self.Hessian(l_vec)) @ self.Jacobian(l_vec)
        return delta_lambda

    def _print_method(self):
        print_IMAXED()

class MAXED(_MAXED_baseclass):
    """
    MAXED unfolding method, inheriting from common DataHandler class.
        Run the MAXED algorithm in Python.
        Replicate the aglorithms used, which are L-BFGS-B(for MC) and Simulated Annealing (FC).
    """
    available_methods = ['simulated_annealing', 'L_BFGS_B', '(custom callable optimizer)'] # where custom refers to using a custom minimization algorithm.
    def __init__(self, *args, desired_chi2=0, **kwargs):
        super().__init__(*args, desired_chi2=desired_chi2, **kwargs)

    def run(self, method='simulated_annealing', *args, **kwargs):
        """
        Run using the chosen method
        """
        assertion_error_string = "The 'method' argument must be either a string (correspond to a method of self) or a callable function (maximization algorithm attempting to maximize the potential)."
        assertion_error_string +="\nThe built-in list of methods are"
        assertion_error_string += str(self.available_methods)
        assert callable(method) or isinstance(method, str), assertion_error_string

        if isinstance(method, str):
            l_vec = getattr(self, method)(*args , **kwargs)
        elif callable(method):
            l_vec = method(self.potential_function, *args, **kwargs) # the user should also declare the initial_l within *args or **kwargs in this use case.
        self._append_lambda(l_vec)

    def L_BFGS_B(self, initial_l=None, *args, **kwargs):
        if initial_l is None:
            initial_l = (np.random.random(self.m)*2 -1)*0.01
        fun = lambda l_vec: -self.potential_function(l_vec) # minimization, not maximization
        jac = lambda l_vec: -self.Jacobian(l_vec) # negative sign follows through to the first order derivative.
        res = minimize(fun, x0=initial_l, method='L-BFGS-B', jac=jac, *args, **kwargs)
        l_vec = res.x
        if self.verbosity>=1:
            print("L-BFGS-B minimization result=\n", res)
        self.lambda_vector = [l_vec,]
        return l_vec

    def simulated_annealing(self, T=1.0, TR=0.85, check_chi2=False, **kwargs):
        potential_function = lambda l: self.potential_function(l)
        if check_chi2:
            check_func = lambda l_vec: self._get_chi2(False, test_phi=self.lambda2spectrum(l_vec))
            self.simann = _SimulatedAnnealer(potential_function, np.zeros(self.m).tolist(), T, TR, maximize=True, verbosity=self.verbosity, check_func=check_func, **kwargs)
        else:
            self.simann = _SimulatedAnnealer(potential_function, np.zeros(self.m).tolist(), T, TR, maximize=True, verbosity=self.verbosity, **kwargs)
        self.simann.run()
        l_vec = self.simann.vector_opt
        return l_vec

    def _print_method(self):
        print_MAXED()

#Fron this point below, everything is made for the IMAXED algorithm.

_NT, _NS = 20, 5 # adjust the step size 20 times per temperature adjustment; and sweep 5 times per such adjustment to gather good enough statistics for the re-calculation of the step size.
# _NEPS = 4 # unused
_MAXEVL = int(1E5)
_ISEED1, _ISEED2 = 1, 2
_bounds = [-1E25, 1E25]
_EPS = 1.0E-6

class _RanNumGenerator():
    '''
    Produces a uniform distribution [0,1]
    '''
    def __init__(self, IJ, KL): # initialize an array of random numbers
        '''
        corresponding SUBROUTINE: RMARIN
        '''
        self.num_calls = 0
        self.U_len = 97
        U = np.zeros(self.U_len, dtype=float32)
        assert 0<=IJ<=31328, "First Seed out of bounds!"
        assert 0<=KL<=30081, "Second Seed out of bounds!"
        # initialize some constants to be used in loops
        C = float32(362436.0 / 16777216.0) # using float 32 to conform to the fortran's REAL format
        CD = float32(7654321.0 / 16777216.0) # using float 32 to conform to the fortran's REAL format
        CM = float32(16777213.0 /16777216.0) # using float 32 to conform to the fortran's REAL format
        I97, J97 = 96, 32
        ii_range, jj_range = 97, 24

        # initialize constants to be used in divisions and additions below.
        d1, d2, d3, d4 = 177, 177, 169, 178
        a1, a2, a3, a4 = 2, 2, 1, 0
        d5 = 179
        m3 = 53
        # initialize the values of ijkl
        i = (IJ//d1)%d2 + a1
        j = (IJ%d1)     + a2
        k = (KL//d3)%d4 + a3
        l = (KL)%d3     + a4
        s_init, t_init = float32(0.0), float32(0.5)

        # perform obfuscation
        for ii in range(ii_range):
            s, t = s_init, t_init
            for jj in range(jj_range):
                #loop to switch numbers around
                m = mod(mod(i*j, d5)*k, d5)
                i = j
                j = k
                k = m
                l = mod(m3*l + a3, d3)
                if mod(l*m, 64)>=32:
                    s += float64(t)
                t *= float32(0.5)
            U[ii] = float32(s)

        for attr, val in zip(['U', 'C', 'CD', 'CM', 'I97', 'J97'], [U, C, CD, CM, I97, J97]):
            setattr(self, attr, val)
            
    def get_num(self):
        self.num_calls += 1
        uni = self.U[self.I97] - self.U[self.J97]
        if uni<0.0:
            uni += float32(1.0)
        self.U[self.I97] = uni
        #move the pointerd down by one row
        self.I97 = (self.I97-1)%self.U_len
        self.J97 = (self.J97-1)%self.U_len
        self.C -= self.CD
        if self.C<0.0:
            self.C += self.CM
        uni -= self.C
        if uni < 0.0:
            uni += float32(1.0)
        return uni

    def __next__(self):
        return self.get_num()

    def ranmar(self): # an alias to conform with the fortran program
        return self.get_num()

    def get_num_scaled(self):
        return self.get_num()*2.0 - 1.0
    
class _BuiltInRanNumGenerator(): # faster
    def __init__(self, seed1, VERSION):
        random.seed(seed1, VERSION)
        self.num_calls=0
    def get_num(self):
        self.num_calls+=1
        return random.random()
    def get_num_scaled(self):
        return self.get_num()*2 - 1

class _SimulatedAnnealer():
    '''
    maximizes the values of the function
    '''
    def __init__(self, objective_func, x, T=1.0, TR=0.85, maximize=False, verbosity=1, rand_gen=_RanNumGenerator, check_func=None, **kwargs):
        '''
        Can use **kwargs to overwrite them.
        '''
        # Since _RanNumGenerator is slower, we ahve opted to use _BuiltInRanNumGenerator instead.
        self.vector = x
        self.temperature = float(T)
        self.T_begin = float(T)
        self.reduction_factor = TR
        assert 0<=self.reduction_factor<1, "Reduction factor must not be negative (temperature must remain positive), and should not be >=1 temperature(should not increase)"
        self.verbosity = verbosity
        self.objective = objective_func # function to be maximized
        self.maximize = maximize
        if not self.maximize:
            self.objective = lambda x: -objective_func(x)
        self.desired_direction = ['Uphill' if self.maximize else 'Downhill' for i in ['dummytext']][0]
        self.undesired_direction= ['Downhill' if self.maximize else 'Uphill' for i in ['dummytext']][0]
        self.n = len(self.vector)
        self.vm = np.ones(self.n, dtype=float) # allowed step size. don't konw what it stands for. Perhaps 'vector movement'?
        #attaching an instance of the rand_gen for this instance of _SimulatedAnnealer to use.
        self.rand_gen = rand_gen(_ISEED1, _ISEED2)

        self.vector_bounds = [np.full(self.n, _bounds[0]), np.full(self.n, _bounds[1])]
        assert self.check_in_bounds(self.vector), "The initial values of the vector must be within these bounds."
        
        # numbers used to record the ups and downs during the optimization process.
        self.Nuphill = self.Nrejected = self.Nnew = self.Ndownhill = self.Naccepted = self.Nout_bounds = self.Nfunc_ev = 0
  # number of:uphill,       rejected,     fresh optima,  downhill,        accepted,       out of bounds,     function evaluations

        self.F_star=[] # the first four are used as dummy values.
        self.NS, self.NT = _NS, _NT

        #take first step
        self.F = self.objective(self.vector)
        assert isinstance(self.F, float), "the objective function must have the signature 'vector input, scalar output'"
        self.F_opt = self.F
        self.vector_opt = self.vector.copy()
        self.Nfunc_ev += 1
        self.acceptance_array = np.zeros(self.n, dtype=int) # loop 320 shrunken down to one line thanks to Python nicety
        self.C_vector = np.full(self.n, 2.0, dtype=float) # it never gets changed
        if self.verbosity>=1:
            print(f'initial value:\nx = {self.vector},\nF = {self.F}')
        self.max_iter = _MAXEVL
        self.EPS = _EPS
        self.upper_adj_limit = 0.6
        self.lower_adj_limit = 0.4
        self.adj_slope = 1/0.4
        self.check_func=check_func
        if kwargs:
            for k,v in kwargs.items():
                setattr(self, k, v)
    def run(self, method='default_run', *args, **kwargs):
        getattr(self, method)(*args, **kwargs)

    def step_size_adjustment(self, ratio, C): # ratio must be between 0 and 1
        if ratio>self.upper_adj_limit:
            return 1 + C*(ratio-self.upper_adj_limit)*self.adj_slope
        elif ratio<self.lower_adj_limit:
            return 1/(1 + C*(self.lower_adj_limit-ratio)*self.adj_slope)
        else:
            return 1

    def default_run(self):
        # loop 100
        while self.Nfunc_ev<=self.max_iter:
            self.Nuphill = self.Nrejected = self.Nnew = self.Ndownhill = self.Nout_bounds = 0
            # loop 400
            for m in range(self.NT):
                self.acceptance_array = np.zeros(self.n, dtype=int) # loop 320 shrunken down to one line thanks to Python nicety
                # loop 300
                for j in range(self.NS): #sweep NS times
                    # loop 200
                    self.take_default_step()
                    #end of loop 200
                # end of loop 300

                # adjust the vm so that approximately half of them will be rejected.
                # loop 310
                scale_factor = [self.step_size_adjustment(ratio, C) for ratio, C in zip(self.acceptance_array/self.NS, self.C_vector)]
                self.vm *= scale_factor
                # end of loop 310
                if self.verbosity>=2:
                    print('Intermediate results after step length adjustment:')
                    print('New step length (vm)=', self.vm)
                    print('Current optimal x=', self.vector_opt)
            # end of loop 400
            if self.verbosity>=1:
                print('Temperature =', self.temperature,'which has been reduced by', len(self.F_star)-1, 'times')
                print('Max function value so far =', self.F_opt)
                print('Number of  trials:')
                print('         total number of moves =',self.Nuphill + self.Ndownhill + self.Nrejected)
                print('{: >30} ='.format(self.desired_direction.lower()), self.Nuphill)
                print('{: >30} ='.format('accepted '+self.undesired_direction.lower()), self.Ndownhill)
                print('{: >30} ='.format('rejected '+self.undesired_direction.lower()), self.Nrejected)
                print('                 out of bounds =', self.Nout_bounds)
                print('new maxima at this temperature =', self.Nnew)
                print('         Current optimal x=', self.vector)
                print('  Current step length (vm)=', self.vm)
                print('')
            # termination condition (loop 410 and 420)
            # if not any( abs(self.F - self.F_star)>EPS ):
            if len(self.F_star)<=4:
                pass
            elif all( abs(self.F - self.F_star[-4:])<=self.EPS ): # if the last four iterations ends with an F similar to this iteration's ending F,
                if (self.F_opt-self.F_star[-4])<=self.EPS: # and we haven't seen a huge gain in the last four iterations

                    if self.check_func is not None: # extra check condition, not implemented in the FORTRAN version.
                        if self.check_func(self.vector_opt)<=self.EPS:
                            if self.verbosity >= 0:
                                print('Simulated Annealling complete (termination criteria achieved).')
                                self.print_end_message()
                            return self.vector_opt
                        else:
                            pass # not accepted yet, keep the loop running

                    else:
                        if self.verbosity >= 0:
                            print('Simulated Annealling complete (termination criteria achieved).')
                            self.print_end_message()
                        return self.vector_opt
            #end of loop 410 and 420

            self.temperature *= self.reduction_factor
            # loop 430
            # self.F_star[1:] = self.F_star[:-1].copy()
            # self.F_star[0] = self.F
            self.F_star.append(self.F.copy())
            # end of loop 430

            self.vector = self.vector_opt.copy()
            # loop 440
            self.F = self.F_opt # substitute in F_opt back as the previous iteration's F.
            # end of loop 440
        # end of loop 100
        print('Maximum number of iteration reached, terminating...')
        self.print_end_message()
        return self.vector_opt

    def print_end_message(self):
        print('The temperature has been reduced by', len(self.F_star), 'times.')
        print('The vector-step size was adjusted', self.NT, 'times per temperature;')
        print('Each vector-step size adjustment required', self.NS, 'sweeps of the vector to gather good enough statistics.')
        print('Each sweep of the vector required changing each bins individually, and taking the step, evaluating the loss function, and conintuing to the next bin, repeated', self.n, 'times')
        print('Which gives a total of', self.Nfunc_ev-1, 'function calls. (+1 initial function call at the starting point)') 
        print('The random number generator was called', self.rand_gen.num_calls, 'times')

    def check_in_bounds(self, value, bin_num=None):
        if bin_num is None:
            return all(self.vector_bounds[0] < value) and all(value < self.vector_bounds[1])
        else:
            return self.vector_bounds[0][bin_num] < value < self.vector_bounds[1][bin_num]

    def take_default_step(self):
        # loop 200
        for h in range(self.n): #sweep across the entire vector once
            #before the start of the iteration, check that we havent 

            vector = self.vector # make a new copy of the vector with the h-th element changed.
            vector[h] += self.vm[h] * self.rand_gen.get_num_scaled() 
            if not self.check_in_bounds(vector[h], bin_num=h):
                Nout_bounds += 1
                vector[h] = self.rand_gen.get_num() * (self.vector_bounds[1][h] - self.vector_bounds[0][h]) # rescale it back into the allowed range.
                if self.verbosity>=3:
                    print(f"Bounds was exceeded while exploring the value for bin {h}, which started from")
                    print('x=', self.vector, 'F=', self.F)
                    print('Instead we will use x=', vector)
            Fp = self.objective(vector)
            self.Nfunc_ev += 1
            if self.verbosity>=3:
                print('old x=', self.vector, 'old F=', self.F)
                print('new x=', vector, 'new F=', Fp)
            
            #uphill case
            if Fp>=self.F:
                self.uphill(vector, Fp, h)
                if self.verbosity>=3: print(self.desired_direction+' step accepted')
            else:
                self.downhill(vector, Fp, h)
        # end of loop 200

    def uphill(self, vector, Fp, bin_changed):
        self.Nuphill += 1
        self.set_vector_and_F(vector, Fp, bin_changed)
        if Fp > self.F_opt: #if new optimum found
            self.F_opt = Fp
            self.Nnew += 1
            self.vector_opt = self.vector.copy()
            if self.verbosity>=3: print('New optimum found.')

    def downhill(self, vector, Fp, bin_changed):
        exponent = (Fp - self.F)/self.temperature # should be -ve.
        p = exp(exponent) # should be a number between zero and one.
        # The smaller abs(exponent) is, the closer to 1 (larger) it should be
        pp = self.rand_gen.get_num()
        if pp<p:
            self.set_vector_and_F(vector, Fp, bin_changed)
            self.Ndownhill += 1
            if self.verbosity>=3: print(self.desired_direction+' step accepted')
        else:
            if self.verbosity>=3: print(self.desired_direction+' step rejected')
            self.Nrejected += 1
            #no more action needed, self.vector[h] will be left as is.

    def set_vector_and_F(self, vector, Fp, bin_changed):
        self.vector = vector
        self.F = Fp
        #record acceptance
        self.Naccepted += 1
        self.acceptance_array[bin_changed] += 1

    def translation_from_umg(self, attr=None):
        '''
        Gets the translated name of the variable in this program.
        Input: fortran name
        Output: python name.
        '''
        umg_to_python_implementation = {
        'x'    : 'vector',
        'TEMP' : 'temperature' ,
        'TEMPR': 'reduction_factor',
        'N'    : 'n',
        'FCN'  : 'objective_func',
        'FP'   : 'Fp',
        'FOPT' : 'F_opt',
        'NS'   : 'NS',
        'NT'   : 'NT',
        'LB'   : 'vector_bounds[0]',
        'UB'   : 'vector_bounds[1]',
        'MAX'  : 'maximize',
        'MAXEVL':'max_iter',
        'VM'   : 'vm',
        'ranmar':'rand_gen.get_num',
        'NUP'  : 'Nuphill',
        'NREJ' : 'Nrejected',
        'NACP' : 'acceptance_array',
        'FSTAR': 'F_star',
        'NNEW' : 'Nnew',
        'NDOWN': 'Ndownhill',
        'NACC' : 'Naccepted',
        'NOBDS': 'Nout_bounds',
        'LNOBDS':'Nout_bounds',
        'NFCNEV':'Nfunc_ev',
        'XOPT' : 'vector_opt',
        'IPRINT':'verbosity',
        'EPS'  : 'EPS',
        }
        if attr is None:
            return umg_to_python_implementation
        else:
            return umg_to_python_implementation[attr]