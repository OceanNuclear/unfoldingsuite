import sys, os, itertools
sys.path.append("..")
from datahandler import UnfoldingDataHandlerLite, UnfoldingDataLoader
from maximumentropy import MAXED, IMAXED, AMAXED
from nonlinearleastsquare import GRAVEL, SAND_II
from regularization import Regularizer
from linearleastsquare import PseudoInverse
from interface import UMG33Interface
from metrics import Metrics
import numpy as np
from numpy import array as ary
from numpy import sqrt
from scipy.stats import norm, lognorm # where log uniform is uniform distribution in log-space.
from matplotlib import pyplot as plt

def get_gs(lower, upper, num_bins):
    """
    Parameters
    ----------
    lower:  lowest energy edge
    upper:  highest energy edge
    num_bins:   total number of bins in the spectrum

    Returns
    -------
    group structure, in 1D
    """
    short_gs = np.logspace(np.log10(lower), np.log10(upper), num_bins+1)
    long_gs = ary([short_gs[:-1], short_gs[1:]]).T
    return short_gs, long_gs

def mono_energetic(beam_mean_E, beam_width_E, beam_intensity, noise_floor, short_gs, beam_cutoffs=None):
    """
    Parameters
    ----------
    beam_mean_E: center line of the gaussian beam's peak.
    beam_width_E: std of the gaussian peak
    beam_intensity: height of the beam intensity
    noise_floor: a noise floor to positively offset the flux values of every bin by.
    short_gs: group structure, given in 1D shape.
    beam_cutoffs: the gaussian beam's effect is limited to a certain range. beam_cutoffs is a tuple/list/array providing the upper and lower limits.

    Returns
    -------
    final_flux: The group-flux resulted by the mono-energetic beam + the noise floor.
    """
    if not beam_cutoffs:
        beam_cutoffs = min(short_gs), max(short_gs)
    signal_cutoffs = norm.ppf(0.01), norm.ppf(0.99)
    def transform_gs_to_signal(energy):
        E_range = np.diff(beam_cutoffs)[0]
        E_min = beam_cutoffs[0]
        S_range = np.diff(signal_cutoffs)[0]
        S_min = signal_cutoffs[0]

        percentage = ary(energy - E_min)/E_range
        S_output = (percentage * S_range) + S_min
        return S_output

    gs_average_E = short_gs[:-1]+np.diff(short_gs)/2 # find the average energy of each bin
    # transform the energy class mark into the signal space.
    gaussian_part = norm.pdf(transform_gs_to_signal(gs_average_E)) * beam_intensity
    final_E_dependent_flux = gaussian_part + noise_floor
    final_flux = np.diff(short_gs) * final_E_dependent_flux
    return final_flux

def plot_spectrum(group_flux, long_gs, ax=None, /, **kwargs):
    if not ax:
        ax = plt.subplot()
    E_dependent_flux = group_flux / (np.diff(long_gs)[:,0]) # turn it into energy-dependent flux (flux per eV) again.
    return ax, ax.semilogx(long_gs.flatten(), np.repeat(E_dependent_flux, 2), **kwargs)[0]

def foil_response(long_gs):
    """
    Create a random cross-section profile, which consists of either a threshold reaction or a thermal-capture+resonance-capture reaction.
    Parameters
    ----------
    long_gs: group structure, given in 2D, each row of which consists of the [lower, upper] class mark.
            (given in eV)
    
    It's hard-coded to randomly generate one of the following two reactions:
    Two kinds of reactions may be produced 
    --------------------------------------
    threshold reaction: the right-hand half of a logit curve + a weakly decreasing slope.
    thermal capture + resonance capture

    Maths
    -------
    microscopic cross-section: given in eV.
    macroscopic cross-section (Sigma): reaction rate cm^-1 = microscopic cross section (sigma, lower case) * Number density
        Number density of solids usually ranges between 1E22 (large atomic number element with low specific gravity) to 1E23 (low atomic number element with high specific gravity)
    
    We can then assume the detector's efficiency = 1.0 count detected per every 10000 reactions, and we are using foils of thickness = 0.1cm.
    And we're using a foil area of 1cm^2.

    This allows us to get a response matrix = 0.00001 times the macroscopic cross-section.
    """

    barns_to_mm = 1E-24
    THRESHOLD_REACTION = np.random.choice([True, False])
    mean_E = long_gs[:, 0] + np.diff(long_gs)[:, 0]/2
    if THRESHOLD_REACTION: # Get a threshold reaction fifty percent of the time.
        peak_height = 10**np.random.uniform(2, 4)
        peak_width = np.random.uniform(1E6, 10E6) # width of the threshold reaction peak
        E_thres = np.random.uniform(0.5E6, long_gs.max())
        def transform_to_lognormal(energy):
            S_range = 1.0
            S_min = 0.0
            percentage = ary(energy - E_thres)/peak_width
            S_output = (percentage * S_range) + S_min
            return np.clip(S_output, 0, None)

        microscopic_xs = lognorm(1).pdf(transform_to_lognormal(mean_E)) * peak_height
    else: # get a normal capture reaction the other fifty percent of the time.
        xs_at_1eV = 10**np.random.uniform(2, 5)
        slope = np.random.uniform(-0.6, -1.2)
        background_decreasing_thermal_capture_xs = xs_at_1eV*mean_E**slope

        resonance = np.zeros(mean_E.shape)
        for i in range(np.random.randint(50)): # add a random number of resonance peaks
            height = np.random.uniform (0, 2*xs_at_1eV)
            center = np.random.uniform(100, 10E3)
            width = np.random.uniform(30, 100)
            signal_cutoffs = norm.ppf(0.01), norm.ppf(0.99)
            def transform_to_res_peak(energy):
                E_min = center - 3*width
                E_range = 6*width
                S_min = norm.ppf(0.01)
                S_range = norm.ppf(0.99) - S_min

                percentage = ary(energy- E_min)/E_range
                S_output = (percentage * S_range) + S_min
                return S_output

            resonance += norm.pdf(transform_to_res_peak(mean_E)) * height
        microscopic_xs = resonance + background_decreasing_thermal_capture_xs
    num_density = np.random.uniform(1E22*barns_to_mm, 1E23*barns_to_mm)
    detection_efficiency = 1/10000 # assume only 1 in 10000 reactions produce detectable count at the detector
    thickness = 0.1 # 1mm thickness
    
    macroscopic_xs = num_density * microscopic_xs
    response = macroscopic_xs * detection_efficiency * thickness
    return response # counts per flux(cm^-2) * cm2, so dimensionless.

def plot_response(response_matrix, long_gs, ax=None, /, **kwargs):
    if not ax:
        ax = plt.subplot()
    for response in response_matrix:
        # ax.semilogx(long_gs.flatten(), np.repeat(response, 2), **kwargs)
        ax.loglog(long_gs.flatten(), np.repeat(response, 2), **kwargs)
    return ax

from _3d_comparison_demo import add_noise

if __name__=='__main__':
    # gs
    min_E, max_E, num_bins = 2, 10E6, 200
    short_gs, long_gs = get_gs(min_E, max_E, num_bins)
    # phi (true)
    true_flux = mono_energetic(3E6, 2E6, 10, 10, short_gs, [1E6, 8E6])
    ax, line = plot_spectrum(true_flux, long_gs)
    ax.set_title("True spectrum")
    plt.show()

    # R (response matrix)
    response_matrix = ary([foil_response(long_gs) for _ in range(20)])
    response_matrix = response_matrix[(response_matrix @ true_flux)>100]
    print("the response matrix has shape=", response_matrix.shape)
    plot_response(response_matrix, long_gs)
    plt.title("Responses from different foils"); plt.show()
    # plot_response(response_matrix[(response_matrix @ true_flux)>100], long_gs)
    # plt.title("selected spectra"); plt.show()

    # N (true)
    N_true = response_matrix @ true_flux
    N_meas = add_noise(N_true)[0]
    covar_N_meas = np.diag(N_meas)
    # phi (a priori)
    apriori = flat_spectrum = mono_energetic(1E6, 1E4, 0, 10, short_gs)

    # short-hand to plot both a priori and truth, and then show the graph to the user, made into a function
    def plot_all_then_show(instance):
        ax = plot_spectrum(instance.solution.phi, long_gs, label="solution")[0]
        plot_spectrum(apriori, long_gs, ax, label="a priori")
        plot_spectrum(true_flux, long_gs, ax, label="truth (initial estimator)")
        plt.legend(); plt.title(str(instance.__class__)); plt.show()

    unf = UnfoldingDataHandlerLite(N_meas=N_meas, apriori=apriori, Rm=response_matrix, covar_N=covar_N_meas, desired_chi2=1.0, verbosity=2)
    unf.group_structure = short_gs

    input("Run programs? (ctrl + c to stop)")

    # reg = Regularizer.copy_from(unf.copy_from(unf, desired_chi2=Regularizer.minimum_desired_chi2)) # can't use minimum_desired_chi2 because that will 
    reg = Regularizer.copy_from(unf.copy_from(unf, desired_chi2=1.0))
    reg.run()
    plot_all_then_show(reg)
    if PLOT_REG:=True:
        fig, ax = plt.subplots()
        ax.plot(reg.tau_records, label="tau"); plt.legend()
        ax.twinx().plot(reg.chi2_val, color="C1", label="chi^2")
        plt.legend(); plt.title("unfolding progress of the regularizer"); plt.xlabel("iteration #"); plt.show()

    amaxed = AMAXED.copy_from(unf.copy_from(unf, desired_chi2=AMAXED.minimum_desired_chi2))
    amaxed.run()
    plot_all_then_show(amaxed)

    imaxed = IMAXED.copy_from(unf.copy_from(unf, desired_chi2=IMAXED.minimum_desired_chi2))
    imaxed.run()
    plot_all_then_show(imaxed)

    ps = PseudoInverse.copy_from(unf.copy_from(unf, desired_chi2=1.0))
    ps.run()
    plot_all_then_show(ps)

    gravel = GRAVEL.copy_from(unf.copy_from(unf, desired_chi2=1.0))
    gravel.run()
    plot_all_then_show(gravel)

    sand2 = SAND_II.copy_from(unf.copy_from(unf, desired_chi2=SAND_II.minimum_desired_chi2))
    sand2.run()
    plot_all_then_show(sand2)