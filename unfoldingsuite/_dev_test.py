# for development purpose. Not for creating results
import pandas as pd
from datahandler import UnfoldingDataHandlerLite, UnfoldingDataLoader
from datahandler import chi2_calculator
from maximumentropy import IMAXED, AMAXED, MAXED
from nonlinearleastsquare import GRAVEL, SAND_II
from interface import UMG33Interface
from regularization import Regularizer
from linearleastsquare import PseudoInverse, GradientDescent, CustomOptimizer
from metrics import Metrics
D_KL = Metrics().D_KL

import os, json
import inspect
from collections import defaultdict
from inspect import signature

# for analysis purpose:
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import array as ary
from numpy import log as ln
from numpy import exp, sqrt
from numpy.linalg import eigvals, inv, norm, det, eig
from numpy.linalg import pinv, inv
mean = lambda vec: vec[:-1]+np.diff(vec)/2
step = lambda vec: plt.step(np.arange(len(vec)), ary(vec), where='post')
norm_step = lambda vec: plt.step(np.arange(len(vec)), ary(vec)/max(vec), where='post')
def easy_logspace(start, stop, *args, **kwargs):
    return np.logspace(np.log10(start), np.log10(stop), *args, **kwargs)

with open('reg_benchmark/unfolding_result_long_backup.json') as f:
    data = json.load(f)
spec_names = list(data.keys())[2:]

os.chdir('UMG3.3_source/check_behaviour3')
from rebin_link import Rebin

Rm = ary(data['response matrix'][:36])
Rm_scaled = ary([ary(line)/sum(line)*(Rm[0].sum()) for line in Rm])
gs_old = data['full group structure']
matrix_rebinner = Rebin(flat_flux_PUL=False)
spectrum_rebinner = Rebin()

rebin_matr = lambda old_matr, trimmed_gs, E_new: ary([matrix_rebinner.re_bin(old_response, trimmed_gs, E_new) for old_response in old_matr])
rebin_spec = lambda old_flux, trimmed_gs, E_new: spectrum_rebinner.re_bin(old_flux, trimmed_gs, E_new)



with open('unfolding_result_short.json') as f:
    result_dict = json.load(f)

m_min, m_max, n_min, n_max = 3, 10, 5, 30
mn_list = ary(np.meshgrid(np.arange(m_min, m_max+1), np.arange(n_min, n_max+1))).T.reshape([-1,2])
def add_noise_poisson(vector, num_copies=3, include_original=True):
    if include_original:
        output_list = [vector.copy()]
        output_list_name = ['original']
    else:
        output_list = []
        output_list_name = []

    for copy in range(num_copies):
        while True:
            sample = ary([np.random.normal(val, sqrt(val)) for val in vector])
            if (sample>0).all():
                output_list.append(sample)
                break
        output_list_name.append('noise'+str(copy+1))
    return output_list, output_list_name

algorithms = {  
                # 'IMAXED':IMAXED,
                # 'AMAXED':AMAXED,
                'MAXED_fc':UMG33Interface,
                # 'MAXED_mc':UMG33Interface,
                'GRAVEL_fc':UMG33Interface,
                # 'MAXED': MAXED,
                # 'Regularization': Regularizer,
                'PseudoInverse': PseudoInverse,
                # 'GradientDescent': GradientDescent,
                'GRAVEL': GRAVEL,
                }
exec_path = {'MAXED_fc': '../check_behaviour/maxed_improved', 'MAXED_mc': '../check_behaviour/maxed_mc', 'GRAVEL_fc': '../check_behaviour/gravel_te'}

# for spec_name in spec_names[:]:
spec_name = spec_names[0]
spectrum = data[spec_name]
trimmed_truth = spectrum['true spectrum']
trimmed_apriori = spectrum['9']['spectra']['A priori']
mask = ary(spectrum['trim mask'], dtype=bool)
trimmed_gs = ary(gs_old)[np.logical_or(mask.tolist()+[False], [False]+mask.tolist())]
trimmed_Rm = ary([line[mask] for line in Rm_scaled])

# for (m,n) in mn_list:
m, n = 3,5
E_new = trimmed_gs[::len(trimmed_gs)//(n+1)][:n+1]
Rm_new = rebin_matr(trimmed_Rm, trimmed_gs, E_new)[:m]
truth = rebin_spec(trimmed_truth, trimmed_gs, E_new)
# make N_meas list
N_meas = Rm_new @ truth
N_perturbations, N_perturbations_name_list = add_noise_poisson(N_meas, num_copies=3)

# make apriori_list
apriori_param = rebin_spec(trimmed_apriori, trimmed_gs, E_new)
apriori_list, apriori_name_list = add_noise_poisson(ary(apriori_param), num_copies=5)
# apriori_flat = np.ones(n)
# apriori_list.append(apriori_flat)

# create the unf object to copy things into
unf = UnfoldingDataHandlerLite(N_meas, Rm_new, apriori_param, sigma_N=sqrt(N_meas), desired_chi2=1.0, verbosity=1)

from unfoldingsuite.nonlinearleastsquare import GRAVEL as GRV, SPUNIT as SPT
grv = GRV()
grv.set_vector('reaction_rates', unf.N_meas.tolist())
grv.set_matrix('response_matrix', unf.Rm.tolist())
grv.set_vector('a_priori', unf.apriori.tolist())
grv.set_vector_uncertainty('reaction_rates', unf.sigma_N.tolist())
grv.set_vector('group_structure', list(range(unf.n + 1)))
grv.run('n_trials', 161)

spt = GRV()
spt.set_vector('reaction_rates', unf.N_meas.tolist())
spt.set_matrix('response_matrix', unf.Rm.tolist())
spt.set_vector('a_priori', unf.apriori.tolist())
spt.set_vector_uncertainty('reaction_rates', unf.sigma_N.tolist())
spt.set_vector('group_structure', list(range(unf.n + 1)))
spt.run('n_trials', 161)

print(chi2_calculator(ary(grv.vectors['reaction_rates']), unf.covar_N_inv, ary(grv.vectors['solution_reaction_rates'])))

for alg_name, alg in algorithms.items():
    alg_instance = alg.copy_from(unf, )
    if alg is UMG33Interface:
        gravel = alg_instance
        gravel.run(alg_name, exec_path[alg_name])
    else:
        alg_instance.run(alg_instance.available_methods[0])
