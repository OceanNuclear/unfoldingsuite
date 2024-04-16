import matplotlib.pyplot as plt
import pandas as pd
import numpy as np; from numpy import log as ln
from scipy.optimize import fsolve, root, Bounds, minimize
from numpy.linalg import inv, det
from numpy import diag, sqrt
import time

#user controlled parameters which speeds up the convergence.
TAU = 1/10000
OVERSHOOT_PROTECTION = 2 #MUST be a number larger than 1, should be larger than 1.188
#user options
VERBOSE = True

'''Calculates the total deviation as tau*cross-entropy(phi, phi_0) + chi^2 over RR'''
#load in the spectra.
apriori = pd.read_csv('a_priori.csv', header=None)
fDEF    = apriori.values.reshape([-1])
answer  = pd.read_csv('answer_spectrum.csv',header=None).values.reshape([-1])
#load in the group structure
# gs      = 
#load in the response matrix
R       = pd.read_csv('R.csv', header=None, index_col=0).values
#reaction rates
rr      = pd.read_csv('reaction_rates.csv', header=None)
N       = rr.values.reshape([-1])
sigma   = 0.05*sqrt(N)
#get the lengths of the reaction rates etc.
n = len(fDEF)
m = len(N)

#make A and c_original
A = np.zeros([n,n])
c_original = np.zeros([n])
c_otherwise = np.zeros([n])
for i in range(n):
    for j in range(n):
        A[i][j] = 2* sum([ R[k][i]*R[k][j]/(sigma[k]**2) for k in range(m) ])
    c_original[i] = TAU * ln(fDEF[i]) + 2* sum([ N[k]*R[k][i]/(sigma[k]**2) for k in range(m) ])
    # c_otherwise[i] = TAU * ln(fDEF[i]-1) + 2* sum([ R[k][i]*N[k]/(sigma[k]**2) for k in range(m) ])

#calculate M and c, which will actually be used in the calculations.
c = np.zeros([n])
c[:-1] = - np.diff(c_original)
c[-1] = 1
M = np.zeros([n,n])
M[:-1] = - np.diff(A, axis=0)
M[-1,:] = 1

#$\matr{L}$
def L_operator(f_g):
    L_vector = ln(f_g)
    first_n1 = - np.diff(L_vector)
    L_g = np.hstack([first_n1,0])
    return L_g

#$\matr{\Lambda}$
def get_Lambda(f_g):
    main_diag = np.diag(1/f_g)
    Lamb_g = np.zeros([n,n])
    Lamb_g[:-1] = - np.diff(main_diag, axis=0)
    return Lamb_g

def get_J(g, tau):
    Lambda_g = get_Lambda(f_g)
    return M + tau*Lambda_g

def get_alpha(delta, f):
    w = delta/f
    possible_undershoots = [abs(w[i]) for i in range(n) if delta[i]<0]
    x = max(possible_undershoots)
    return np.clip(1/(OVERSHOOT_PROTECTION*x), 0,1)

def get_relative_entropy(P, Q):
    return sum(-P*ln(Q/P))

def get_chisq(f,N):
    N_prime = R.dot(f)
    chi2 = (N-N_prime)**2/(sigma**2)
    return sum(chi2)

def plot_det_J_vs_tau(f_g):
    taus = np.logspace(-5,0)
    dets = np.array([det(get_J(f_g,t)) for t in taus])
    displayable = np.clip(abs(dets), 1e-280,1e280)==abs(dets)
    # plottable = np.logical_and(displayable, dets!=0)
    plt.loglog(taus[displayable],dets[displayable],marker='x')
    plt.show()
    return

#Start iterating
f_g = fDEF #f0 = fDEF
#logging information
iteration=0
chi2=[]
D_KL=[]
while True:
    #dummy variable.
    iteration+=1
    if VERBOSE: print("iteration =",iteration)
    f_g1 = f_g
    
    #Evaluate the RHS into a single vector.
    RHS = c-M.dot(f_g1) - TAU*L_operator(f_g1) #haven't forced the last line of the equation to equal zero yet.
    
    #step to get delta
    matr_J_g1 = get_J(f_g1, TAU)
        # print(matr_J_g1)
    if VERBOSE: print("det(J) =", det(matr_J_g1))
    delta_g1 = inv(matr_J_g1).dot(RHS)

    #increment by delta * underrelaxation constant
    alpha_g1 = get_alpha(delta_g1, f_g1)
    if VERBOSE: print("alpha =", alpha_g1)
    f_g = f_g1 + alpha_g1*delta_g1

    #log the two metrics
    chi2.append(get_chisq(f_g,N))
    D_KL.append(get_relative_entropy(f_g,fDEF))

    #pause program, continue at user's discretion.
    options = input("type in g to graph, e to show entropy, X to show chi^2, v to turn off VERBOSE, c to break...")
    if "g" in options:
        plt.plot(f_g, label="program solution")
        plt.plot(fDEF, label="a priori")
        plt.plot(answer, label="true solution")
        plt.legend()
        plt.show()
    if "p" in options:
        plot_det_J_vs_tau(f_g1)
    if "v" in options:
        VERBOSE = not VERBOSE
    if "e" in options:
        print("entropy =", D_KL[-1])
    if "X" in options:
        print("chi^2/DoF =", chi2[-1])
    if "c" in options:
        break
    print()