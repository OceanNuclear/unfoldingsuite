import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import sqrt
from numpy import log as ln
# from scipy.optimize import fsolve, root, Bounds, minimize
import time
from pynverse import inversefunc

ACCURACY = 6
TAU = 1/10000


#user controlled parameters which speeds up the convergence.
TAU = 5/1000
OVERSHOOT_PROTECTION = 2 #MUST be a number larger than 1, should be larger than 1.188
#user options
VERBOSE = True

'''Calculates the total deviation as tau*cross-entropy(phi, phi_0) + chi^2 over RR'''
#load in the spectra.
apriori = pd.read_csv('a_priori.csv', header=None)
fDEF    = apriori.values.reshape([-1])
answer  = pd.read_csv('answer_spectrum.csv',header=None).values.reshape([-1])
#load in the response matrix
R       = pd.read_csv('R.csv', header=None, index_col=0).values
#reaction rates
rr      = pd.read_csv('reaction_rates.csv', header=None)
N       = rr.values.reshape([-1])
sigma   = 0.05*sqrt(N)
#get the lengths of the reaction rates etc.
n = len(fDEF)
m = len(N)

#Creating the intermediate constants which will be reused multiple times.
A = np.zeros([n,n])
const = np.zeros([n])
# solution = np.zeros([n])
for i in range(n):
    for j in range(n):
        A[i][j] = 2* sum([ R[k][i]*R[k][j]/(sigma[k]**2) for k in range(m) ])
    const[i] = 2* sum([ R[k][i]*N[k]/(sigma[k]**2) for k in range(m) ])
c = const - TAU * (ln(fDEF)-1)

detA = np.linalg.det(A) #note that this is zero.
    #Should print a lot of nan's
    #This is as expected because there should be multiple solutions.

# #Cramer's rule: for more visual solving of the set of linear equations.
# def cramer(c, matrix = A):
#     for j in range(n+1):
#         A_prime =  A.copy()
#         A_prime[:,j] = c
#         print(np.linalg.det(A_prime)) #note that this will also be zero.
#         solution[j] = np.linalg.det(A_prime)/detA
#         print(solution[j])
#     return solution

def get_mu(f):
    mu_vector = A.dot(f) + TAU*ln(f) - c
    return mu_vector

