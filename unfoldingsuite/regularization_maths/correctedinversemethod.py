import matplotlib.pyplot as plt
import pandas as pd
import numpy as np; from numpy import log as ln
from numpy.linalg import inv, det
import time

#user controlled parameters which speeds up the convergence.
TAU = 1E-7

def quadrature(l):
    return np.sqrt(sum([i**2 for i in l])) 

class BareRegularizer:
    def __init__(self, N, S_N, R, apriori, tau=TAU):
        self.N = N
        self.R = R
        self.S_N = S_N # THE INVERSE of the covariance matrix! (i.e. inv( np.diag( [std[i]**2 for i in range(m)] ) )) == np.diag([1/(std[i]**2) for i in range(m)]) because that's what we need
        self.fDEF = np.array(apriori)/sum(apriori)
        self.n = len(self.fDEF) #get the number of bins
        self.m = len(N) #get the number of reactions
        assert R.shape == (self.m,self.n), f"Expected the response matrix R to be an np.ndarray of shape ({self.m},{self.n})."
        assert S_N.shape == (self.m,self.m), f"Expected the covariance matrix for N = S_N to be a square np.ndarray with side-length {self.n}"
        self.tau = tau
        self.alpha = 0.5 # DO NOT put alpha>=1.
        for attr in ['Y','f','w', 'steepness', 'chi2_val', 'reg_val', 'loss_val']:
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
        return -self.tau * self.fDEF/f  # return a len n vector

    def get_df(self, Y, f):
        RHS = self.tau * self.fDEF/f
        RHS -= self.mu(Y, f) * np.ones(self.n)
        RHS -= self.grad_chi(Y, f)
        # pre_multiplier_vector = (Y* self.R @ f -self.N) @ self.S_N @ (Y*self.R)
        # RHS = self.tau*(self.fDEF/f - 1) + ( pre_multiplier_vector @ f ) - (pre_multiplier_vector)
        hessian_matrix = ( Y* self.R.T ) @ self.S_N @ (Y * self.R)
        hessian_matrix += self.tau * np.diag(self.fDEF/(f**2))
        inverse_matrix = inv(hessian_matrix)
        return inverse_matrix.dot(RHS) # return a len n vector

    def step_size(self, f, df):
        negatives = [ df[i]/f[i] for i in range(self.n) if df[i]<0]
        if len(negatives)>0:
            x = abs(min(negatives))
            w = min([self.alpha/x, 1])
        else:
            w = 1
        return w
    
    def chi2_func(self, Y, f):
        difference_vec = (Y*self.R) @ f - self.N
        chi2 = difference_vec @ self.S_N @ difference_vec
        return chi2

    def reg_func(self, f):
        return self.fDEF.dot( ln(self.fDEF)-ln(f) )

    def loss_func(self, Y, f):
        return self.reg_func(f) + self.chi2_func(Y, f)

    def take_step(self):
        #update Y
        self.Y.append( self.best_fit_n_yield(self.f[-1]) ) #get the latest iteration of Y
        df = self.get_df( self.Y[-1], self.f[-1]) #get the full step vector
        
        #calculate losses according to the previous iteration's f plus this new iteration's f
        self.chi2_val.append( self.chi2_func(self.Y[-1], self.f[-1]) )
        self.reg_val.append( self.reg_func(self.f[-1]) )
        self.loss_val.append( self.loss_func(self.Y[-1], self.f[-1]) )

        #Record how far we're going to step
        self.w.append( self.step_size(self.f[-1], df) ) # get the multiplier for the full step
        f_ = self.f[-1] + self.w[-1] * df 
        if self.active_normalization:
            self.f.append(f_/sum(f_))
        else:
            self.f.append(f_) #take a step equal to multiplier * full step vector
        if len([i for i in self.f[-1] if i<=0])>0:
            assert False, "WE HAVE A PROBLEM, we have a nonpositive in f[-1]" 
        gradient_vector = self.grad_reg(self.f[-1])+ self.grad_chi(self.Y[-1], self.f[-1])+ self.tau
        # Note the very interseting result that, at the true minimum, we have grad(loss) = - tau *np.ones(n).
        # This is because we're restricting ||f||_1 = 1.
        # i.e. we force sum(f)=1
        # To make sure that we get 0*np.ones(n) when we're at the origin, self.tau is added on the end.
        self.steepness.append( quadrature(gradient_vector) )

    def set_alpha(self, alpha):
        assert 0<alpha<1, f"magnitude of {alpha=} out of range!"
        if alpha>0.5:
            print(f"Warning: {alpha=}>0.5 may lead to an oscillatory approach to the minimum due to overrelaxation.")
        self.alpha = alpha

    def copy(self):
        # Copys only the initialization variables
        return BareRegularizer(self.N, self.S_N, self.R, self.fDEF)

if __name__=="__main__":
    apriori = pd.read_csv('a_priori.csv', header=None)
    answer  = pd.read_csv('answer_spectrum.csv',header=None).values.reshape([-1])
    #load in the group structure
    # gs      = 
    #load in the response matrix
    R       = pd.read_csv('R.csv', header=None, index_col=0)
    #reaction rates
    rr      = pd.read_csv('reaction_rates.csv', header=None)
    sigma   = np.sqrt(rr.values.reshape([-1]))

    reg = BareRegularizer(rr.values.reshape([-1]), np.diag(1/sigma**2), R.values, apriori.values.reshape([-1]))
    reg.take_step()
    reg.active_normalization = True
    
    iteration=0
    while iteration<30:
        iteration+=1
        # plt.plot(reg.f[-1])
        # plt.show()
        print(f"{iteration=}")
        # print(f"{sum(reg.f[-1])=}")
        reg.take_step()