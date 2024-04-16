import matplotlib.pyplot as plt
import pandas as pd
import numpy as np; from numpy import log as ln
from scipy.optimize import fsolve, root, Bounds, minimize
from numpy.linalg import inv, det, svd
from numpy import diag, sqrt
import time

#user controlled parameters which speeds up the convergence.
TAU = 1E-5

# num_tol= np.finfo(TAU).max #largest number that the computer can store safely
apriori = pd.read_csv('a_priori.csv', header=None)
answer  = pd.read_csv('answer_spectrum.csv',header=None).values.reshape([-1])
#load in the group structure
# gs      = 
#load in the response matrix
R       = pd.read_csv('R.csv', header=None, index_col=0)
#reaction rates
rr      = pd.read_csv('reaction_rates.csv', header=None)
sigma   = sqrt(rr.values.reshape([-1]))

def quadrature(l):
    return sqrt(sum([i**2 for i in l])) 

class Regularizer:
    def __init__(self, N, S_N, R, fDEF, tau=TAU):
        self.N = N
        self.R = R
        self.S_N = S_N
        self.fDEF = fDEF/sum(fDEF)
        self.n = len(self.fDEF) #get the number of bins
        self.m = len(N) #get the number of reactions
        assert R.shape == (self.m,self.n), f"Expected the response matrix R to be an np.ndarray of shape ({self.m},{self.n})."
        assert S_N.shape == (self.m,self.m), f"Expected the covariance matrix for N = S_N to be a square np.ndarray with side-length {self.n}"
        self.tau = tau
        self.alpha = 1/2 # DO NOT put alpha>=1.
        for attr in ['Y','f','w']:
            setattr(self,attr,[])
        #start the iteration from fDEF
        self.f.append(self.fDEF) 
        self.active_normalization = True

    def best_fit_n_yield(self, f):
        right_vector_len_m = self.S_N @ self.R @ f
        return (self.N @ right_vector_len_m)/( self.R @ f @ right_vector_len_m) #return a scalar

    def mu(self, Y, f):
        return self.tau - self.grad_chi(Y, f) @ f # returns a scalar

    def grad_chi(self, Y, f):
        N_ = Y * self.R @ f
        return (N_ - self.N) @ self.S_N @ (Y * self.R) # return a len n vector

    def grad_reg(self, f):
        return -self.tau * self.fDEF/f # return a len n vector

    def get_df(self, Y, f):
        RHS = -self.grad_reg(f)
        RHS -= self.mu(Y, f) * np.ones(self.n)
        RHS -= self.grad_chi(Y, f)
        LHS_matrix = ( Y* self.R.T ) @ self.S_N @ (Y * self.R)
        LHS_matrix += self.tau * np.diag(self.fDEF/(f**2))
        inverse_matrix = inv(LHS_matrix)
        return inverse_matrix.dot(RHS) # return a len n vector

    def step_size(self, f, df):
        negatives = [ df[i]/f[i] for i in range(self.n) if df[i]<0]
        if len(negatives)>0:
            x = abs(min(negatives))
            w = min([self.alpha/x, 1])
        else:
            w = 1
        return w
    
    def iter(self):
        self.Y.append( self.best_fit_n_yield(self.f[-1]) ) #get the latest iteration of Y
        df = self.get_df( self.Y[-1], self.f[-1]) #get the full step vector
        self.w.append( self.step_size(self.f[-1], df) ) # get the multiplier for the full step
        if self.active_normalization:
            self.f.append( (self.f[-1])/sum(self.f[-1]) + self.w[-1]*df ) #take a step equal to multiplier * full step vector
        else:
            self.f.append( (self.f[-1]) + self.w[-1]*df ) #take a step equal to multiplier * full step vector
        if len([i for i in self.f[-1] if i<0])>0:
            assert False, "SHIT WE HAVE A PROBLEM, we have a negative in f[-1]" 
    
    def copy(self):
        print("Copying only the initialization variables")
        return Regularizer(self.N, self.S_N, self.R, self.fDEF)

reg = Regularizer(rr.values.reshape([-1]), np.diag(1/sigma**2), R.values, apriori.values.reshape([-1]))
reg.iter()
reg.active_normalization = False
if __name__=="__main__":
    while True:
        # plt.plot(reg.f[-1])
        # plt.show()
        print(f"{sum(reg.f[-1])=}")
        print(f"{reg.Y[-1]=}")
        reg.iter()