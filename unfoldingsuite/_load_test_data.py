import pandas as pd
from .datahandler import UnfoldingDataHandlerLite, UnfoldingDataLoader
from .datahandler import chi2_calculator
from .maximumentropy import IMAXED, AMAXED, MAXED
from .nonlinearleastsquare import GRAVEL, SAND_II
from .interface import UMG33Interface
from .regularization import Regularizer
from .linearleastsquare import PseudoInverse, GradientDescent, CustomOptimizer
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
# MAKE IT EXPANDABLE, CLOPEN! reduce the number of places I have to change in order to add a new alg

for spec_name in spec_names[:]:
    spectrum = data[spec_name]
    trimmed_truth = spectrum['true spectrum']
    trimmed_apriori = spectrum['9']['spectra']['A priori']
    mask = ary(spectrum['trim mask'], dtype=bool)
    trimmed_gs = ary(gs_old)[np.logical_or(mask.tolist()+[False], [False]+mask.tolist())]
    trimmed_Rm = ary([line[mask] for line in Rm_scaled])

    # result_dict[spec_name] = {}
    for (m,n) in mn_list:
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
        unf = UnfoldingDataHandlerLite(N_meas, Rm_new, apriori_param, sigma_N=sqrt(N_meas), desired_chi2=np.clip(0,m-n,None), verbosity=1)
        
        for alg_name, alg in algorithms.items():
            alg_instance = alg.copy_from(unf, desired_chi2=alg.minimum_desired_chi2)
            if alg is UMG33Interface:
                def run_alg_instance():
                    alg_instance.run(alg_name, exec_path[alg_name])
            else:
                def run_alg_instance():
                    alg_instance.run(alg_instance.available_methods[0])

            # create plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title( alg_name+' with matrix shape = ({},{})'.format(m, n))

            for (apriori, apriori_name) in zip(apriori_list, apriori_name_list):
                D_kl_apriori = D_KL(apriori, truth)
                unf.apriori = apriori

                for N_pert, N_pert_name in zip(N_perturbations, N_perturbations_name_list):
                    unf.N_meas = N_pert

                    line = [] # create empty array to record the where the line goes
                    for desired_chi2 in np.linspace(300, max([alg.minimum_desired_chi2, m-n])):
                        unf.desired_chi2 = desired_chi2
                        alg_instance = alg.copy_from(unf)
                        try:
                            run_alg_instance()
                        except Exception as e:
                            print(e)
                            pass # do nothing, we'll let it be
                        print("Finished one run of algorithm={}, desired_chi2={}".format(alg_name, desired_chi2))
                        line.append([D_kl_apriori, alg_instance.solution.chi2, D_KL(alg_instance.solution.phi, truth)])
                        # D_KL(alg_instance.solution.phi, apriori)
                    plt.plot(*ary(line).T, label='a priori={}, N_meas={}'.format(apriori_name, N_pert_name))
            ax.set_xlabel('D_KL(apriori, truth)')
            ax.set_ylabel('solution chi2')
            ax.set_zlabel('D_KL(solution, truth)')
            # plt.legend()
            plt.show()
            input()

import sys
sys.exit()

"""
N_meas = pd.read_csv('../tests/testdata/5ChannelReactionRates.csv', header=None).values.flatten()
sigma_N = pd.read_csv('../tests/testdata/5ChannelReactionRateUncertainties.csv', header=None).values.flatten() * N_meas
Rm = pd.read_csv('../tests/testdata/5ChannelResponse.csv', header=None).values
apriori = pd.read_csv('../tests/testdata/5ChannelAPriori.csv', header=None).values.flatten()

truth = pd.read_csv('../tests/testdata/5ChannelTrueFlux.csv', header=None).values.flatten()
"""

"""
N_meas = pd.read_csv('../example/example_data_set_2/7_NIF_Ignition_reaction_rates.csv', header=None).values.flatten()
sigma_N = pd.read_csv('../example/example_data_set_2/7_NIF_Ignition_reaction_rate_uncertainties.csv', header=None).values.flatten() * N_meas
Rm = pd.read_csv('../example/example_data_set_2/7_NIF_Ignition_ACT_response_matrix.csv', header=None).values
apriori = pd.read_csv('../example/example_data_set_2/7_NIF_Ignition_a_priori_spectrum.csv', header=None).values.flatten()

truth = pd.read_csv('../example/example_data_set_2/7_NIF_Ignition_true_flux_spectrum.csv', header=None).values.flatten()
"""
"""
np.random.seed(0)

m, n = 3, 6
truth = ary([5,2,3,4,2,3,1,3,4])[:n] # use a real rebinned fusion neutron specturm instead
Rm = np.random.random(100).reshape([10,10])[:m, :n] * 2 # also rebin from some existing nuclear data libraries of cross section

N_meas = Rm @ truth
sigma_N = sqrt(N_meas)
apriori = np.ones(10)[:n]

unf = UnfoldingDataLoader()
unf.sigma_N = sigma_N
unf.N_meas = N_meas
unf.Rm = Rm
unf.apriori = apriori
unf.desired_chi2 = 0
"""