from .datahandler import UnfoldingDataHandlerLite
from .graphics import print_PSEUDO_INVERSE, print_GRADIENT_DESCENT, print_CUSTOM_OPTIMIZER

from numpy import array as ary
from numpy import log as ln
from numpy import exp
import numpy as np
import time, copy
from numpy.linalg import pinv
from collections import deque

class SlicableDeque(deque):
    def __getitem__(self, ind):
        '''
        Increase the capability of the deque so that I can take slices out for plotting.
        This is only done for the pseudoinverse module because no other modules would require as many steps.
        '''
        if type(ind)==slice:
            return list(self)[ind]
        else:
            return super().__getitem__(ind)

def _get_scale_factors(phi):
    """
    returns a list of scale factors of len==len(phi),
    so that not any((scale_factor * phi)>1).
    """
    ratio_to_highest = phi/max(phi) # other scaling function than this may be chosen.
    return ratio_to_highest # a 1d vector

class PseudoInverse(UnfoldingDataHandlerLite):
    """
    Algorithm to takes steps towards the nearest bottom of the valley.
    """
    available_methods = ["direct_step", "restrict_near_zero_bases", "limit_fractional_change",]
    minimum_desired_chi2 = 1E-25
    def __init__(self, *args, desired_chi2 = minimum_desired_chi2, conv_speed=1.0, max_iter=np.inf, deletion_limit:int = 5E5, **kwargs):
        """
        conv_speed controls the speed of descent towards the minimum.
        The SlicableDeque's will only store the last deletion_limit elements. 
        stops the algorithm after it has taken max_iter steps.
        """
        super().__init__(*args, desired_chi2=desired_chi2, **kwargs)
        self.Rm_inv = pinv(self.Rm)

        # Use SlicableDeque instead for the following lists of quantities
        # turn new lists for recording data into deques
        self.step_size = SlicableDeque(self.step_size)
        self.phi = SlicableDeque([self.apriori])
        self.N = SlicableDeque([self.Rm @ self.apriori])
        self.chi2_val = SlicableDeque([self._get_chi2(append=False)])

        if self.verbosity>=2:
            print('Overwriting the self.phi, self.N, self.chi2_val, and self.step_size lists as SlicableDeques.')
            if self.verbosity>=3:
                print('Be careful that, if self.Rm, self.apriori, and the covariance on the measurements were changed after initialization')
                print('without changing self.phi, self.N, self.chi2_val, self.step_size back to Deques, we will have a problem when self.num_steps > deletion_limit.')

        self.conv_speed = conv_speed # speed of convergence
        self.max_iter = max_iter
        self.deletion_limit = int(deletion_limit)

        """
        # debugging/development tools. Used only for evaluating the effectiveness.
        self._debugging = bool(DEBUGGING)
        if self._debugging:
            self._desired_step = SlicableDeque([])
            self._scale_factors = SlicableDeque([])
        """

        # using an int to track the number of steps to save time, so that we don't have to keep finding len(deque).
        self.num_steps = 0
        # Also tracks number of steps even after the deletion_limit is reached.

    def take_step(self, method, **kwargs):
        """
        Take one iteration, using the method of choice.
        """
        diff_phi = getattr(self, method)(self.N_meas - self.N[-1], self.phi[-1], **kwargs)
        final_phi = self.conv_speed * diff_phi + self.phi[-1]
        self._append_phi(final_phi)
        self.num_steps += 1
        self.step_size.append(self.conv_speed)

    def direct_step(self, diff_N, current_phi):
        """
        It so happens that some terms cancel when performing the direct step,
        leaving a very simple expression as follows.
        """
        return self.Rm_inv @ diff_N

    def restrict_near_zero_bases(self, diff_N, current_phi):
        """
        restrict the movement in the axes which are close to zeros.
        e.g. for the spectrum phi=[0.1, 1,2],
            we will only allow the first bin to take 5% of its desired step,
            only allow the second bin to take 50% its desired step,
            and allow the third bin to take 100% of its desired step.
        """
        scale_fac = _get_scale_factors(current_phi)
        return scale_fac * self.direct_step(diff_N, current_phi)

    def limit_fractional_change(self, diff_N, current_phi, max_abs_frac_dev=np.inf):
        """
        max_abs_frac_dev control how big of a direct step to be plugged into the scaling formula.
        """
        direct_step = self.direct_step(diff_N, current_phi)
        fractional_change = direct_step/current_phi
        if any(abs(fractional_change)>max_abs_frac_dev):
            fractional_change *= max_abs_frac_dev/abs(fractional_change)
        #scaling formula:
        desired_dest = exp(fractional_change)*current_phi # other functions than exp() may be chosen.
        return desired_dest - current_phi

    def run(self, method=available_methods[0]):
        """
        use the user-chosen method to run pseudoinverse.
        """
        if self.verbosity>=2: print("Using", method, "to perform pseudo-inverse unfolding...")
        if self.verbosity>=1: starttime = time.time()
        while self.chi2_val[-1]>self.desired_chi2 and self.num_steps<self.max_iter:
            self.take_step(method)
            if self.num_steps>self.deletion_limit:
                self.N.popleft()
                self.phi.popleft()
                self.chi2_val.popleft()
                self.step_size.popleft() # pop the Deque to stop memory overflowing
        if self.verbosity>=1:
            print("Unfolding completed in {:2.2f}s after taking {} steps".format(time.time()-starttime, self.num_steps))
        final_phi, step_size_taken = self._interpolate(self.desired_chi2, output_x=True)
        if self.verbosity>=2:
            print("Interpolated between the final two iterations to get a more accurate fit to the desired chi^2.")
        self._append_phi(final_phi)
        self.step_size.append(-1+step_size_taken) # backtracked by this amount in order to get the final solution.

    def _print_method(self):
        print_PSEUDO_INVERSE()

class CustomOptimizer(UnfoldingDataHandlerLite):
    minimum_desired_chi2 = 1E-25
    def __init__(self, *args, desired_chi2=None, **kwargs):
        super().__init__(*args, desired_chi2=desired_chi2, **kwargs)
        self.Rm_inv = pinv(self.Rm)

        self.loss_func= lambda phi: self._get_chi2(False, phi)
        self.Jacobian = lambda phi: 2 * self.covar_phi_inv(phi-(self.Rm_inv @ self.N_meas)) # gradient vector
        self.Hessian= lambda phi: self.covar_phi_inv # Hessian (i.e. second order derivative) matrix
        if self.verbosity>=3:
            print("When using scipy.optimize methods, use the following arguments:")
            print("fun = self.loss_func")
            print("x0 = self.x0")
            print("jac = self.Jacobian")
            print("hess = self.Hessian")

    @property # x0
    def x0(self):
        return self.apriori # link apriori
    @x0.setter
    def x0(self, new_starting_spectrum):
        self.apriori = new_starting_spectrum

    def _print_method(self):
        print_CUSTOM_OPTIMIZER()

class GradientDescent(UnfoldingDataHandlerLite):
    """
    Consists of two methods:
    # one is gradient descent with intelligently calculated step size,
    run(intelligent_step_size=True)

    # the other is momentum-accelerated gradient descent.
    run(intelligent_step_size=False) # momentum is by default 1.

    # one can specify the momentum and decay_rate, and adjust the starting learning rate as well. For example,
    self.learning_rate = 0.001 # increase learning rate from default of 1E-5 to 1E-3.
    run(intelligent_step_size=False, momentum=0, decay_rate=0) # where momentum=0, decay_rate=0 is the same as naive gradient descent.
    """
    available_methods = [True, False] #intelligent_step_size = boolean
    minimum_desired_chi2 = 1E-18
    def __init__(self, *args, desired_chi2=minimum_desired_chi2, learning_rate=1E-5, **kwargs):
        """
        Variation in Rm is not allowed.
        """
        super().__init__(*args, desired_chi2=desired_chi2, **kwargs) # ignore the Rm_error
        self._append_phi(self.apriori)
        self.Rm_inv = pinv(self.Rm)
        self.learning_rate = learning_rate
        self.prev_gradient = self.Jacobian.copy()
        self.step_length = []

    @property
    def Hessian(self):
        return self.covar_phi_inv

    @property
    def Jacobian(self):
        """
        Jacobian vector = grad vector of the chi2 function.
        """
        return 2 * self.covar_phi_inv @ ( self.solution.phi - (self.Rm_inv@self.N_meas) )

    def calculate_step_size(self, direction):
        """ Calculates the step size that will lead to THE minimum in that direction. """
        grad = direction
        step_size = (grad @ self.Rm.T @ self.covar_N_inv @ ((self.Rm@self.solution.phi) - (self.N_meas))) / (grad @ self.covar_phi_inv @ grad)
        return step_size

    def take_step(self, momentum=0.9, decay_rate=1E-5):
        """
        Take a step downhill.
        momentum (beta) should take 0<=beta<1
        decay_rate modifies the learning_rate -> learning_rate/(1+decay_rate+num_step)
        According to Andrew Ng (Coursera) momentum=0.9 is pretty good.
        """
        #Calculate the step
        grad = self.Jacobian
        step = (1-momentum)*grad + momentum*self.prev_gradient
        #Calculate the step size
        step_size = self.learning_rate/(1 + decay_rate*len(self.phi))
        #record the gradient
        self.prev_gradient = grad.copy()
        #take the actual step
        self._append_phi(self.phi[-1] - step_size*step)
        self.step_size.append(step_size)
        self.step_length.append(np.linalg.norm(step))

    def run(self, intelligent_step_size=True, **kwargs): # momentum=0.9, decay_rate=0.1 by default.
        """

        Walk downhill and reach below the desired_chi2.
        use intelligent_step_size to choose between the two methods available (see the documentation for this class.)
        """
        assert isinstance(intelligent_step_size, bool), "intelligent_step_size must be True or False."
        if self.verbosity>=1: starttime = time.time()
        if intelligent_step_size:
            while self.solution.chi2>self.desired_chi2:
                # Using momentum=0, decay_rate=0, take the largest step down hill until we hit the bottom in that direction, and then change course.
                self.learning_rate = self.calculate_step_size(self.Jacobian)
                self.take_step(0, 0)
            description = "with intelligently calculated step sizes (momentum=0, decay_rate=0)."
        else:
            while self.solution.chi2>self.desired_chi2:
                self.take_step(**kwargs)
            if self.verbosity>=1:
                import inspect
                take_step_args = inspect.getfullargspec(self.take_step)
                description = "with kwargs " + str(kwargs) + " overriding the default values of "\
                                + str(take_step_args.args[1:]) + " = " + str(take_step_args.defaults) + "."
        final_phi, step_size = self._interpolate(self.desired_chi2, output_x=True)
        self._append_phi(final_phi)
        self.step_size.append(-1+step_size)
        if self.verbosity>=1:
            print("Unfolding completed in {:2.2f}s after taking {} steps".format(time.time()-starttime, len(self.chi2_val) ))
            print("Optimization completed using gradient descent,", description)

    def _print_method(self):
        print_GRADIENT_DESCENT()

# RMSprop is the next tier up; might be too difficult to implement.
# class RMSprop(GradientDescent):
#     pass