import numpy as np
from numpy import array as ary

class Rebin():
    def __init__(self, flat_flux_PUL=True):
        self.flat_flux_PUL = flat_flux_PUL # assume the flux is evenly distributed within each bin in lethargy space instead of energy space

    def re_bin(self, flux_old, e_bins_old, e_bins_new):
        assert all(np.diff(e_bins_new)>0), "Energy group boundaries must be in ascending order."
        assert all(np.diff(e_bins_old)>0), "Energy group boundaries must be in ascending order."
        assert ary(e_bins_old).ndim == 1, "Must have a linear structure"
        assert ary(e_bins_new).ndim == 1, "Must have a linear structure"
        flux_old = ary(flux_old).flatten() # make sure it's a 1D array
        out_matrix = []
        if self.flat_flux_PUL:
            for sourceL, sourceH in zip(e_bins_old[:-1],e_bins_old[1:]):
                total_leth_space = np.log(sourceH)-np.log(sourceL)
                dest_space = np.diff(np.log(np.clip(e_bins_new, sourceL, sourceH)))
                # if an error/warning is raised at this point, it is likely due to presence of a bin boundary with energy E<=0.
                out_matrix.append(dest_space/total_leth_space)
                # this line should sum to unity.
                # It describes probability of neutron in e_bins_old[i] landing in e_bins_new[j] after rebinning
        else: # assume flat flux in energy space instead
            for sourceL, sourceH in zip(e_bins_old[:-1],e_bins_old[1:]):
                total_spacing = sourceH - sourceL
                dest_space = np.diff(np.clip(e_bins_new, sourceL, sourceH))
                out_matrix.append(dest_space/total_spacing) # this line should sum to unity too
        flux_new = ary(out_matrix).T @ flux_old
        return flux_new.tolist()