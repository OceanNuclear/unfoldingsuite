import matplotlib.pyplot as plt
import pandas as pd
import numpy as np; from numpy import log as ln
from scipy.optimize import fsolve, root, Bounds, minimize
import time

TAU = 0.001

'''Calculates the total deviation as tau*cross-entropy(phi, phi_0) + chi^2 over RR'''
apriori = pd.read_csv('a_priori.csv', header=None)
rr = pd.read_csv('reaction_rates.csv', header=None)
answer = pd.read_csv('answer_spectrum.csv',header=None).values.reshape([-1])

R  = pd.read_csv('R.csv', header=None, index_col=0).values
N = rr.values.reshape([-1])
sigma = 0.05+(N-N)
fDEF = apriori.values.reshape([-1])
n = len(fDEF)
m = len(N)

c = np.zeros([n-1])
D = np.zeros([n-1,n-1])
for i in range(n-1):
    # print(i)
    c[i] = -2/TAU * sum([ N[k]/(sigma[k]**(-2)) * (R[k][i]-R[k][n-1]) for k in range(m)]) - ln(fDEF[i]) + ln(fDEF[n-1])#remember to use [n-1] instead of [n] due to python indexing
    for j in range(n-1):
        D[i][j]= 2/TAU * sum([ (R[k][i]-R[k][n-1])*(R[k][j]-R[k][n-1])/(sigma[k]**2)  for k in range(m) ])#remember to use [n-1] instead of [n] due to python indexing
#solving with method 1
def set_of_equations_results(x):
    f = x/sum(x)
    equation = np.zeros([n])
    for i in range (n-1):
        equation[i] += sum([ D[i][j]for j in range(n-1) ]) 
        equation[i] += c[i]
        equation[i] += ln(f[i]) 
        equation[i] -= ln( f[n-1]) # N.B. DO NOT USE 1-sum([ f[j] for j in range(n-1)]) as this will lead to a very small number but with very low precision, thus is stored as 0.000 on the computer instead.
    equation[n-1] = sum([ f[j] for j in range(n)]) -1
    return equation

def lagrangianminimizationresult(f):
    nmu=np.zeros(n)
    for i in range(1,10):
        nmu[i]+=2*sum([ f[j]*sum([ (R[k][i]*R[k][j])/(sigma[k]**2) for k in range(m) ]) for j in range(n)])
        nmu[i]+=TAU*ln(f[i])
        nmu[i]-=TAU*ln(fDEF[i])
        nmu[i]+=TAU
        nmu[i]-=2*sum([ (R[k][i]*N[k])/(sigma[k]**2) for k in range(m) ])
    return [ nmu[i+1]-nmu[i] for i in range(n-1) ] + [ sum([f[j] for j in range(n) ])-1 ]
# bound = Bounds(np.zeros(n), np.ones(n))
# def min_set(f):
#     return sum([abs(i) for i in set_of_equations_results(f)])
x = root(set_of_equations_results, x0=fDEF)
# x1 = root(set_of_equations_results,x0=fDEF, method='lm')  #Seems to give an answer that's sufficiently close, even though it's not right.
# x2 = root(lagrangianminimizationresult,x0=fDEF, method='broyden1')
# x3 = root(lagrangianminimizationresult,x0=fDEF, method='broyden2')
print(x)
#'linearmixing' will get out of bounds,
#'krylov' gives an answer that's identical to the starting. (i.e. failure)
#minimize(min_set,...) will also lead to an answer that's identical to the starting, but instead it declares it not a failure.
#'hybr' leads to the same problem, i.e. an answer that's identical to the starting.
#'broyden1' also leads to identical answer as the starting.
#Perhaps the regularization constant TAU=1 is just too big?

#########################################
if False:
    #tau = 0 case: i.e. do not apply any regularization information.
    A = np.zeros([n+1,n+1])
    e = np.zeros([n+1])
    solution = np.zeros([n+1])
    for i in range(n):
        e[i] = 2* sum([ R[k][i]*N[k]/(sigma[k]**2) for k in range(m) ])
        for j in range(n):
            A[i][j] = 2* sum([ R[k][i]*R[k][j]/(sigma[k]**2) for k in range(m) ])
    A[:,-1] = 1
    A[-1,:] = 1
    A[-1,-1]= 0
    e[-1] = 0
    #Cramer's rule:
    detA = np.linalg.det(A) #note that this is zero.
    for j in range(n+1):
        A_prime =  A.copy()
        A_prime[:,j] = e
        print(np.linalg.det(A_prime)) #note that this will also be zero.
        solution[j] = np.linalg.det(A_prime)/detA
        print(solution[j])
        #Should print a lot of nan's
        #This is as expected because there should be multiple solutions.