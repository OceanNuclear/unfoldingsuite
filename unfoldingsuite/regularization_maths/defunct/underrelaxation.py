import matplotlib.pyplot as plt
import pandas as pd
import numpy as np; from numpy import log as ln
from scipy.optimize import fsolve, root, Bounds, minimize
from numpy.linalg import inv, det, svd
from numpy import diag, sqrt
import warnings

#user controlled parameters which speeds up the convergence.
TAU = 1E-7
OVERSHOOT_PROTECTION = 2 #MUST be a number larger than 1, should be larger than 1.255
#user options
VERBOSE = True

num_tol= np.finfo(TAU).max #largest number that the computer can store safely
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

#Create the values here.
A = np.zeros([n,n])
c_original = np.zeros([n])
c_otherwise = np.zeros([n])
for i in range(n):
    for j in range(n):
        A[i][j] = 2* sum([ R[k][i]*R[k][j]/(sigma[k]**2) for k in range(m) ])
    c_original[i] = TAU * ln(fDEF[i]) + 2* sum([ N[k]*R[k][i]/(sigma[k]**2) for k in range(m) ])
    # c_otherwise[i] = TAU * ln(fDEF[i]-1) + 2* sum([ R[k][i]*N[k]/(sigma[k]**2) for k in range(m) ])

# def analyze_svd(A):
#     tol = 1E-7
#     A_svd_left, magnitude, A_svd_right = np.linalg.svd()
#     non_zeros = magnitude>tol
#     plt.plot(A_svd_left[non_zeros]) #seems to give a graph simplar
#     plt.title("The left svd vectors")
#     plt.show()

#     Q = R.T/sigma
#     W = Q.T - Q.mean(axis=1)
#     W = W.T
#     plt.plot( W /abs(W).max(axis=0) ) #gives something that LOOKS like the left svd above.
#     return

# def check_if_spanned_by(u,v): #This is WRONG unless v is a list of orthogonal vector.
    # import pytest
    # assert len(u)==len(v[0]), "Expecting u to be a vector and a v to be a list of vectors. They should have matching shapes"
    # u,v = np.array(u)/root_sum_squared(u), np.array(v) # make sure that they are both np.array's #make u a unit vector
    # resid = u.copy()
    # for vi in v:
        # vi = vi/root_sum_squared(vi)
        # resid -= u.dot(vi)* vi
    # print(f"u Is spanned by v: {resid==pytest.approx(0)}")
    # return resid

#calculate M and c, which will actually be used in the calculations.

c = np.zeros([n])
c[:-1] = - np.diff(c_original)
c[-1] = 1

#Check that all singular values are +ve:
lsv, svd_values, rsv = svd(A)
try:
    assert (svd_values[:m]>0).all, "Expected a matrix contructed from sums of m linearly independent outerproducts to have at least m non-zero singular values."
except AssertionError:
    warnings.warn("May get multiple stationary points, i.e. can converge on a local minimum/maximum instead of a stationary point.")

def valid_tau_range(f_g):
    #walk to find out the allowed range of tau's
    taus = np.linspace(10, -10) #Hard-coded range of possible exponents of 10 to try.
    detJ = lambda t: np.log(det(get_J(f_g,10.0**t)))
    np.seterr('ignore') #immediately set it back to print warning because we know it WILL lead to a floating point error below.
    range_of_taus = [t for t in taus if np.isfinite(defJ(t))] #range of allowed taus
    np.seterr('print')
    
    det_J_upper = lambda t: np.log10(det(get_J(f_g, 10.0**t))) - np.log10(num_tol) #calculate how close it is to the smallest computable number by the program in log space.
    det_J_lower = lambda t: np.log10(det(get_J(f_g, 10.0**t))) + np.log10(num_tol) #calculate how close it is to the smallest computable number by the program in log space.

    max_exp = fsolve(det_J_upper, max(range_of_taus))
    min_exp = fsolve(det_J_lower, min(range_of_taus))

    return 10.0**max_exp, 10.0**min_exp

# print(f"initial numerically stable range of tau values ={valid_tau_range(fDEF)}")

def get_M(Y=1):
    M = np.zeros([n,n])
    M[:-1] = - np.diff(A, axis=0)
    M[-1,:] = 1
    return M
M = get_M()

#\vector{f} Dependent operators:
#r"$\matr{L}$"
def L_operator(f_g):
    L_vector = ln(f_g)
    first_n1 = - np.diff(L_vector)
    L_g = np.hstack([first_n1,0])
    return L_g

#r"$\matr{\Lambda}$"
def get_Lambda(f_g):
    main_diag = np.diag(1/f_g) # Change the main_diagonal
    Lamb_g = np.zeros([n,n]) #Lambda
    Lamb_g[:-1] = - np.diff(main_diag, axis=0) #first 1 to (n-1) rows
    # leave the last row as zeros.
    return Lamb_g

def get_J(f_g, tau):
    Lambda_g = get_Lambda(f_g)
    # if CALIBRATE_N_YIELD: M = get_M(get_y_yield(f_g))
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

def plot_deviation_vs_tau(f_g):
    taus = np.logspace(-5,0)
    neg_dev, pos_dev, rms_dev = [], [], []
    iden = np.identity(len(f_g))
    for t in taus:
        J = get_J(f_g,t)
        matrix = J.dot(inv(J)) #multiply by the inverse itself
        flat_dev = (matrix-iden).flatten()
        rms_dev.append( root_sum_squared(flat_dev) )
        pos_dev = sum([ i for i in flat_dev if i>0])
        neg_dev = sum([ i for i in flat_dev if i<0])
    plt.semilogx(taus,pos_dev,label='positive deviation')
    plt.semilogx(taus,neg_dev,label='negative deviation')
    plt.semilogx(taus,rms_dev,label="rms deviation")
    plt.show()
    return
'''
def plot_det_J_vs_tau(f_g):
    taus = np.logspace(-5,0)
    dets = np.array([det(get_J(f_g,t)) for t in taus])
    displayable = np.clip(abs(dets), 1e-280,1e280)==abs(dets)
    # plottable = np.logical_and(displayable, dets!=0)
    plt.loglog(taus[displayable],dets[displayable],marker='x')
    plt.show()
    return
'''
def get_O_of_set_of_equations(f_g):
    # if CALIBRATE_N_YIELD: M = get_M(get_y_yield(f_g))
    residuals = M.dot(f_g) - c
    L_vector = ln(f_g)
    residuals[:-1] += TAU * - np.diff(L_vector)
    return residuals

def get_n_yield(f_g):
    Q = np.apply_along_axis(np.divide,0,R,sigma) #each row is scaled by 1/sigma

    # R_over_sigma_dot_f = Q.dot(f_g) #len = k
    # R_over_sigma_dot_f.dot(R_over_sigma_dot_f) # self dot-product, over all k elements
    denominator = f_g.dot((A/2).dot(f_g)) 
    
    kth_coef = np.apply_along_axis(np.multiply, 0, Q, N/sigma)
    vectorized_coefs = np.sum(kth_coef, axis=0)
    numinator = vectorized_coefs.dot(f_g)

    return numinator/denominator

def root_sum_squared(vector):
    return sqrt(sum([i**2 for i in vector]))

# def neutron_spec_plot(*args, **kwargs):
#     return plt.plot(args, kwargs)

#Start iterating
f_g = fDEF #f0 = fDEF
#logging information
iteration=0
chi2=[] # record of chi^2
D_KL=[] # record of relative entropy
alpha=[]# record of step size
O   =[] # record of deviation from the origin in all dimensions (len=n vector)
O_dist=[] # record of distance from the origin (scalar).
CALIBRATE_N_YIELD = False

while True:
    iteration+=1
    if VERBOSE: print("iteration =",iteration)
    #dummy variable.
    f_g1 = f_g #f_g1 represents r"$f^{g-1}$"
    
    #Evaluate the RHS into a single vector.
    # if CALIBRATE_N_YIELD: M = get_M(get_y_yield(f_g1))
    RHS = c-M.dot(f_g1) - TAU*L_operator(f_g1) #haven't forced the last line of the equation to equal zero yet.
    
    #step to get delta
    matr_J_g1 = get_J(f_g1, TAU)
        # print(matr_J_g1)
    if VERBOSE: print("det(J) =", det(matr_J_g1))
    delta_g1 = inv(matr_J_g1).dot(RHS)

    #increment by delta * underrelaxation constant
    alpha_g1 = get_alpha(delta_g1, f_g1)
    alpha.append(alpha_g1)
    if VERBOSE: print("alpha =", alpha_g1)
    f_g = f_g1 + alpha_g1*delta_g1

    #calculate the best fit neutron yield
    Y = get_n_yield(f_g)
    if VERBOSE: print(f"Y={Y}")
    #log the two metrics
    chi2.append(get_chisq(f_g,N))
    D_KL.append(get_relative_entropy(f_g,fDEF))

    #Record the deviation from the origin in the codomain space (i.e. how much gradient is left).
    O.append(get_O_of_set_of_equations(f_g))
    O_dist.append( root_sum_squared(O[-1]) )
    #pause program, continue at user's discretion.
    options = input("type in g to graph, v to turn off VERBOSE, G to break and plot everything...")
    if "g" in options:
        # plt.plot(f_g, label="program solution")
        # plt.plot(fDEF, label="a priori")
        # plt.plot(answer, label="true solution")
        plt.plot()
        plt.legend()
        plt.show()
    if "p" in options:
        plot_det_J_vs_tau(f_g1)
    if "v" in options:
        VERBOSE = not VERBOSE
    # if "e" in options:
    #     print("entropy =", D_KL[-1])
    # if "X" in options:
    #     print("chi^2/DoF =", chi2[-1])
    if "G" in options:
        plt.plot( D_KL,chi2, marker='+')
        plt.xlabel("Kullback-Leibler Divergence")
        plt.ylabel(r"$\chi^2$")
        plt.scatter(D_KL[-1],chi2[-1], marker='o')
        plt.show()
        plt.plot(alpha)
    
        plt.title("Step size taken")
        plt.xlabel("Iterations")
        plt.ylabel(r"$\alpha$")
        plt.show()
        
        # for g in range(len(O)):
        #     plt.plot(O[g])
        #     plt.title(f"Deviation in codomain space over {g} iterations")
        #     plt.show()
        # also plot the variation of neutron yield

        plt.title("The distance from the origin in codomain space")
        plt.xlabel("Iterations")
        plt.ylabel(r"$L^2$ distance")
        plt.plot(np.log10(O_dist), label="log10 of the value")
        plt.plot(np.diff(np.log10(O_dist)))
        plt.show()
        break
    print()
#Can stop program when distance from origin in codomain space is smaller than threshold;
#after a check that alpha=1 and stability check (no large oscillation on the chi2 D_KL graph and no large oscillations in the O_g graph) has been performed.
#I don't 