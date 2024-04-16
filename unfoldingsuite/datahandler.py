from .graphics import print_UNFOLDING_DATA_HANDLER_LITE

import numpy as np
from numpy import array as ary
from numpy import sqrt
from numpy.linalg import inv
import json, csv
# UnfoldingDataHandlerLite and SolutionExtractor are crated by me (Ocean) on 2020-07-10 11:57:27.
# The other classes are legacy code from Ross Worrall (UKAEA)

"""
verbosity explanation:
1: information for beginners
2: messages for debugging purpose, tracking the program's progress, as well as making the output pretty (using) _print_method()
3: hints for advanced programmers
0: For advanced users when batch programming, to silence all outputs and ignore all error messages if possible.

All Unfolding algorithm classes can be ran by
        #instantiation:
        # can provide all required data rigth away, or load them in later.
        unfolder = UnfoldingDataHandlerLite(...)
        # optional: load/reload the data separately, if not all of the required data is provided at run-time.
        unfolder.load('quantity', 'filename.file-extension') 
        unfolder.load('quantity', some_list)

        # process
        unfolder.run(unfolder.available_methods[0])

        # results
        # optional: save them to file
        unfolder.solution.quantity.save('filename.file-extension')

        # copying
        unfolder2 = SpecificUnfolder.copy_from(unfolder)
        unfolder2.desried_chi2 = new_chi2 
        unfolder2.run(unfolder2.available_methods[0])

the available methods are listed in SpecificUnfolder.available_methods
"""

class SolutionExtractor():
    """
    Pointer to refer to the solution values in the outerclass,
    which are usually the last values in each of the list attributes.
    """
    def __init__(self, outerclass):
        self.outerclass = outerclass # save the pointer to its own outerclass.
        if not issubclass(type(self.outerclass), UnfoldingDataLoader):
            raise TypeError("Only allow UnfoldingDataLoader or UnfoldingDataHandlerLite instances to be used.")

    # use @property (getters) as pointers # only need getters
    @property # N 
    def N(self):
        return self.outerclass.N[-1]

    @property # phi
    def phi(self):
        return self.outerclass.phi[-1]

    @property # chi2
    def chi2(self):
        return self.outerclass.chi2_val[-1]

    @property
    def covar_phi(self):
        covar_phi_inv = self.outerclass.Rm.T @ self.outerclass.covar_N_inv @ self.outerclass.Rm
        if np.linalg.matrix_rank(covar_phi_inv)==np.shape(covar_phi_inv)[0]:
            return inv(covar_phi_inv)
        else:
            # return np.linalg.pinv(covar_phi_inv) # actually this is wrong!
            raise np.linalg.LinAlgError("inverse of covariance matrix is singular. Please use (self.covar_phi_inv) instead of (self.solution.covar_phi)")

    def print(self, attr):
        """
        print a scalar/vector/vector
        """
        print(getattr(self, attr))

    def save(self, attr, filename, fmt=None):
        """
        Save a scalar, vector, or matrix quantity.
        infer the save file format from the filename's extension if no fmt is provided
        """
        delimiter_dict = {'csv':',', 'ssv':' ', 'tsv':'\t', 'json':None}
        if fmt is None: # infer
            fmt = filename.split('.')[-1] 
        if fmt not in delimiter_dict.keys():
            raise KeyError("The only allowed options for 'fmt' are :" + str(delimiter_dict.keys())+"which are comma-, space-, or tab-separated files, and json." )

        values = getattr(self, attr) # get the values
        if fmt=='json': #take care of the json case
            with open(filename) as f:
                json.dump(ary(values).tolist(), f)
        else: # take care of the csv, ssv, tsv cases.
            if np.ndim(values)>2:
                raise NotImplementedError("Saving 3D or above tensors is not implemented yet, due to the lack of a universally agreed text represnetation. Try using json instead.")
            values.to_file(filename, sep=delimiter_dict[fmt])

def chi2_calculator(N1, covar_N_inv, N2):
    '''
    A simpler expression to calculate chi2, without having to use @.
    Plug in the inverse of the np.diag(sigma**2) matrix as covar_N_inv
    '''
    return (N1-N2)@covar_N_inv@(N1-N2)

class UnfoldingDataLoader():
    """
    This class checks the shape of vector/scalar/matrices listed in the default arguments in the __init__ signature.
    """
    available_methods = []
    minimum_desired_chi2 = 0.0
    def __init__(self, N_meas=None, Rm=None, apriori=None,
                sigma_N=None, covar_N=None,
                desired_chi2=None,
                verbosity:int=1):
        """
        relevant shapes:
        self.n: len(self.Rm[0]), len(self.apriori)
        self.m: len(self.Rm), len(self.N_meas), len(self.sigma_N), len(self.apriori)
        scalar: desired_chi2, verbosity
        any other properties: not set.
        """
        self.verbosity = int(verbosity) # accepts integer between 0 to 3

        if N_meas is not None:
            self.N_meas = N_meas
        if Rm is not None:
            self.Rm = Rm
        if apriori is not None:
            self.apriori = apriori
        # self.m is set when setting N_meas
        # self.n is set when setting Rm
        # self.N, and self.phi are also created when setting apriori
        
        # error values
        if (covar_N is not None) or (sigma_N is not None):
            if covar_N is not None:
                if sigma_N is not None and self.verbosity>=1: # ignore the input for sigma_N if covar_N is provided
                    print("covar_N overrides sigma_N and is used to set the error on N_meas.")
                self.covar_N = covar_N
            else:
                self.sigma_N = sigma_N

        if desired_chi2 is not None:
            self.desired_chi2 = desired_chi2 #desired chi^2 at the end of the unfolding

        # other lists used for recording step sizes and errors.
        self.step_size = []
        self.chi2_val = []
        self.N = []

        if self.verbosity>=2:
            self._print_method()

    def load(self, attr, filename):
        """
        load a scalar, vector, or matrix quantity from file.
        The file can consist of comma-, space- or tab-separated values.
        It can also be a json file.
        """
        with open(filename) as f:
            first_char = f.read(1)
            f.seek(0) # go back

            if first_char in ['{', '[']: # take care of the json case
                setattr(self, attr, ary(json.load(f)))
                return
            else:
                text = f.read()
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(text)
            delimiter = None if dialect.delimiter == ' ' else dialect.delimiter # using None allows for merging of space delimiters.
            header = sniffer.has_header(text)
        except: # if the sniffer can't determine what type of delimiter is used.
            if self.verbosity>=1:
                print("Unable to determine the delimtier used, and the presence of header rows in {}".format(filename))
                print("Parsing 'None' as the delimiter argument to np.genfromtxt, and assuming no header...")
            header = 0
            delimiter = None

        data = np.genfromtxt(filename, delimiter=delimiter, autostrip=True, skip_header=int(header))
        
        if self.verbosity>=2:
            print("Loaded data from file {} with shape {}".format(filename, data.shape))

        setattr(self, attr, data)

    def _check_set_m(self, value):
        if hasattr(self, 'm'):
            if self.m != value: raise ValueError("Input does not match the required lenght m = {}".format(self.m))
        else:
            self.m = value

    def _check_set_n(self, value):
        if hasattr(self, 'n'):
            if self.n != value: raise ValueError("Input does not match the required lenght = n {}".format(self.n))
        else:
            self.n = value

    @property # N_meas
    def N_meas(self):
        return self._N_meas
    @N_meas.setter
    def N_meas(self, vector_of_N_meas):
        assert np.ndim(vector_of_N_meas)==1, "Must be a vector"
        self._check_set_m(len(vector_of_N_meas))
        self._N_meas = ary(vector_of_N_meas)
    @N_meas.deleter
    def N_meas(self):
        del self._N_meas

    @property # sigma_N (meas) 
    def sigma_N(self):
        return self._sigma_N
    @sigma_N.setter
    def sigma_N(self, sigmas):
        assert np.ndim(sigmas)==1, "This must be a vector"
        self._check_set_m(len(sigmas))
        self.covar_N = np.diag(sigmas)**2
        # self._sigma_N = sigmas
        # matrix = np.diag(sigmas**2)
        # self._covar_N = matrix
        # self._covar_N_inv = inv(matrix)

    @sigma_N.deleter
    def sigma_N(self):
        del self._sigma_N
        del self.covar_N

    @property # covar_N (meas) # the _meas is omitted so that it doesn't become too long
    def covar_N(self):
        return self._covar_N
    @covar_N.setter
    def covar_N(self, matrix):
        assert np.ndim(matrix)==2, "expected matrix"
        assert np.shape(matrix)[0] == np.shape(matrix)[1], "Expected square matrix"
        self._check_set_m(np.shape(matrix)[0])
        self._covar_N = matrix
        self._sigma_N = sqrt(np.diag(matrix))
        self._covar_N_inv = inv(matrix)
    @covar_N.deleter
    def covar_N(self):
        del self._covar_N
        del self._covar_N_inv

    @property # covar_N_inv # inverse of the covariance matrix: inversion is a intensive process, would be nice to have a copy of it kept at all times.
    def covar_N_inv(self):
        return self._covar_N_inv

    @property # covar_phi_inv # inverse of the covariance matrix of the spectrum
    def covar_phi_inv(self):
        if hasattr(self, 'Rm_list'):
            Rm = self.Rm_list[-1]
            return Rm.T @ self._covar_N_inv @ Rm
        else:
            return self._covar_phi_inv # saved as a variable, so we don't have to perform this calculation every time we use the getter.
            # this would be set at the __init__ method for the respective algorithms after Rm is set, because it is dependent on the value of Rm

    @property # Rm
    def Rm(self):
        if hasattr(self, 'Rm_list'):
            return self.Rm_list[-1]
        elif hasattr(self, '_Rm'):
            return self._Rm
    @Rm.setter
    def Rm(self, mat):
        # set the self.m and self.n if needed.
        assert np.ndim(mat)==2, "Expected a matrix."
        self._check_set_m(np.shape(mat)[0])
        self._check_set_n(np.shape(mat)[1])
        self._Rm = ary(mat)
    @Rm.deleter
    def Rm(self, mat):
        del self._Rm

    """ # self.Rm_error is not implemented as most classes do not use it.
    @property # Rm_error
    def Rm_error(self):
        if not hasattr(self, '_Rm_error'):
            self._Rm_error = None
        return self._Rm_error # else return None
    @Rm_error.setter
    def Rm_error(self, tensor):
        if tensor is not None:
            assert np.ndim(tensor)>2, "A tensor of at least 3D is expected"
            test_m, test_n = np.shape(tensor)[-2:]
            self._check_m(test_m)
            self._check_n(test_n)
            assert all([i in (self.m, self.n) for i in set(np.shape(tensor))]), "Each dimension must have lenght that matches eitehr self.m or self.n"
        self._Rm_error = tensor

    @Rm_error.deleter
    def Rm_error(self):
        del self._Rm_error
    """

    @property # apriori
    def apriori(self):
        return self._apriori
    @apriori.setter
    def apriori(self, apriori_vector):
        assert np.ndim(apriori_vector)==1, "Expected a vector"
        self._check_set_n(len(apriori_vector))
        self._apriori = ary(apriori_vector)
    @apriori.deleter
    def apriori(self):
        del self._apriori

    @property # desired_chi2
    def desired_chi2(self):
        """
        The new funcitonality is just to return the set value of _desired_chi2.
        The old functionality was to 'Give the default value if None is found', but this is commented away.
        """
        # if self._desired_chi2 is None:
        #     if self.verbosity>=1:
        #         print("Warning: no desired_chi2 value is supplied.")
        #         print("This isn't supposed to happen unless a desired_chi2 default argument is missing in our source code's signature;")
        #         print("or if a custom method is used.")
        #         print("Using the default value (=D.o.F. for overdetermined, 0 otherwise) instead...")
        #     self._desired_chi2 = np.clip(self.m - self.n, self.minimum_desired_chi2, None)
        return self._desired_chi2
    @desired_chi2.setter
    def desired_chi2(self, value):
        """
        Make sure the desired_chi2 a valid value
        """
        if value<self.minimum_desired_chi2:
            raise ValueError("desired value of chi^2 is too low for this algorithm. It must be non-negative, and must be non-zero for some iterative solvers.")
        elif value<(self.m - self.n): # only triggered in overdetermiend case
            if self.verbosity>=1:
                print("Warning: this is an overdetermined case, where chi^2 is expected to equal to (self.m-self.n)={}.".format(self.m-self.n))
                print("Using desired_chi2={} is below the recommended value; It is possible that no such solution exists.".format(value))
        self._desired_chi2 = value
    @desired_chi2.deleter
    def desired_chi2(self):
        del self._desired_chi2

    def _print_method(self):
        print_UNFOLDING_DATA_HANDLER_LITE()

class UnfoldingDataHandlerLite(UnfoldingDataLoader):
    """
    Add methods to load and set the input data,
    as well as append and pop lists.
    """
    _essential_attr_list = ['N_meas', 'Rm', 'apriori', 'covar_N', 'desired_chi2']
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for attr in self._essential_attr_list:
            if not hasattr(self, '_'+attr):
                raise AttributeError("{} is not provided!".format(attr))
        self.phi = []
        self.N = []
        self.chi2_val = []
        self._covar_phi_inv = self.Rm.T @ self.covar_N_inv @ self.Rm
        self.solution = SolutionExtractor(self)
        # self.check_all_present() # bodge needed to make the current (unedited) UnfoldingDataHandlerLite work with the existing classes

    def _get_chi2(self, append=False, test_phi=None):
        """
        The most common use case for _get_chi2 is to append a value of chi2.
        The second most common use case for _get_chi2 are when I have a new flux that I would like to test it's chi2,
        or when I would like to append the last chi2 value into the chi2_val list.
        """
        chi2_this_iteration = chi2_calculator(self.N_meas, self.covar_N_inv, self.N[-1] if test_phi is None else (self.Rm@test_phi))
        if self.verbosity>=2:
            if self.verbosity>=3:
                print("Comparing the self.N[-1] against self.N_meas")
            print("Chi-squared value =", chi2_this_iteration)
        if append:
            self.chi2_val.append(chi2_this_iteration)
        else:
            return chi2_this_iteration

    def _interpolate(self, target_chi2, output_x=False, ind=[-2, -1]):
        r"""
        interpolate (or extrapolate) between two spectra to choose one that gives exactly the right chi^2.
        These two spectra are the spectra pointed to by self.phi[ind[0]] and self.phi[ind[1]].
        Taking advantage of the fact that any straight line traversing through the phi space will always experience a quadratic change in chi^2.
        This fact is obvious from the fact that the second derivative of the following equation (d^2/dx^2) is always a constant.
        \chi^2 (phi0+x*dphi) = (N_meas - Rm@phi0 - x*Rm@dphi)@covar_N_inv@(N_meas - Rm@phi0 - x*Rm@dphi)
        """
        ind0, ind1 = ind # unpack indices
        phi0, phi1 = self.phi[ind0], self.phi[ind1] # take out the corresponding two entries of phi
        N0 = self.N_meas - (self.Rm @ phi0) # difference between the anchor's would-have-been N and the actual N
        dphi = phi1-phi0 # step size
        dN = self.Rm @ dphi # difference between the would-have-been N of the anchor vs the step
        a = (dN @ self.covar_N_inv @ dN)
        b = (N0 @ self.covar_N_inv @ dN)*(-2)
        c = (N0 @ self.covar_N_inv @ N0) - target_chi2
        root = np.roots([a, b, c])
        chosen = np.argmin(abs(root))
        x = root[chosen]
        if self.verbosity>=2:
            print("Back tracking by interpolating between self.phi[{}] and self.phi[{}],".format(ind0, ind1))
            print("which has chi^2={}, {} respectively,".format(self.chi2_val[ind0], self.chi2_val[ind1]))
            print("to reach chi^2={}.".format(target_chi2))
        if output_x:
            return (phi0 + x *  dphi, x)
        else:
            return phi0 + x * dphi

    def _append_phi(self, phi):
        """
        convenient method to record phi, N, and chi^2 value all at once.
        """
        self.phi.append(phi.copy())
        self.N.append(self.Rm@phi)
        self._get_chi2(append=True)

    def _pop_phi(self):
        """
        Used for removing unwanted iterations so that we can back-track to the optimal iteration.
        """
        self.phi.pop()
        self.N.pop()
        self.chi2_val.pop()

    def _print_method(self):
        print_UNFOLDING_DATA_HANDLER_LITE()

    @classmethod
    def copy_from(cls, unf, *args, **kwargs):
        """
        Creates a new instance UnfoldingDataHandlerLite (OR THE CHILD-CLASS that inherits from UnfoldingDataHandlerLite)
        using an instance of UnfoldingDataHandlerLite (OR an instance of ANY OTHER CHILD-CLASS).
        Can overwrite existing attribute when copying using the kwargs
        """
        if not issubclass(type(unf), UnfoldingDataLoader):
            raise TypeError("This method is used for copying info from one instance into another.")

        new_kwargs = {attr: getattr(unf, attr) for attr in cls._essential_attr_list if hasattr(unf, '_'+attr)}

        # verbosity value copied into the new instance.
        new_kwargs.update({'verbosity': unf.verbosity})

        # manual inputs overrides the rest
        new_kwargs.update(kwargs)

        return cls(**new_kwargs)