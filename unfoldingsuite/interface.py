from .datahandler import UnfoldingDataHandlerLite, SolutionExtractor, UnfoldingDataLoader
from .graphics import print_UMG3_3_INTERFACE

from numpy import array as ary
from numpy import log as ln
from numpy import sqrt, exp
import numpy as np
from decimal import Decimal, InvalidOperation
import subprocess
import os
from datetime import datetime, date
import time

"""
Python user -> (interface) -> UMG 3.3 program input -> UMG 3.3 computation
Python user <- (interface) <- UMG 3.3 program input <- UMG 3.3 result
The Python user always input energy values in eV
Everything else uses MeV
"""

def _modified_array2string(array):
    return np.array2string(array, max_line_width=80).replace('[', ' ').replace(']', '\n')

class SolutionExtractor_UMG(SolutionExtractor):
    @property
    def chi2(self):
        return float(super().chi2)

class SolutionExtractor_UMG_MAXED(SolutionExtractor_UMG):
    @property
    def lambda_vector(self):
        return [float(i) for i in self.outerclass.lambda_vector[-1]]

    @property
    def covar_phi_inv(self):
        covar_phi_inv = self.outerclass.get_covar_phi_inv(self.lambda_vector)
        
        if np.linalg.matrix_rank(covar_phi_inv)==np.shape(covar_phi_inv)[0]:
            return np.linalg.inv(covar_phi_inv)
        else:
            return np.linalg.pinv(covar_phi_inv)

class UMG33Interface(UnfoldingDataHandlerLite):
    """
    Write the data into a format usable by UMG 3.3, and use the command line to run it as well.
    available methods for writing into UMG include the following:
    available_methods = ['MAXED_fc', 'MAXED_mc', 'GRAVEL_fc', 'GRAVEL_mc']
    """
    available_methods = ['MAXED_fc', 'MAXED_mc', 'GRAVEL_fc', 'GRAVEL_mc']
    def __init__(self, *args, desired_chi2=None, apriori_error=None, group_structure=None, max_iter=40000, **kwargs):
        """
        self.dummy_measurement_width is only used when multichannel unfolding.
        """
        super().__init__(*args, desired_chi2=desired_chi2, **kwargs)

        if group_structure is not None:
            self.group_structure = group_structure
            if self.verbosity>=1:
                print("Group structure (i.e. list of edges of the bins) is assumed to be inputted with units = 'eV'.")
        else:
            self.group_structure = np.arange(self.n+1)
            if self.verbosity>=1:
                print('Using a dummy group structure = np.arange(self.n+1)')
        if apriori_error is None:
            # apriori_error is used as the default flux INSTEAD when there are bins with negative flux values in the apriori.
            # Hopefully if the inputs are correct, this should never have to be used.
            # self.apriori_error = sqrt(apriori)
            self.apriori_error = abs(self.apriori)
        else:
            self.apriori_error = apriori_error # must be a list with the same len as apriori.
            # apriori_error is only needed if any element inside apriori takes a negative value. In that case UMG will use apriori_error[i] instead of apriori[i]

        self.max_iter = max_iter
        self.dummy_measurement_width = 1 # eV

        self.solution = SolutionExtractor_UMG(self)

    @property # group_structure
    def group_structure(self):
        return self._group_structure
    @group_structure.setter
    def group_structure(self, vector):
        """
        Authors note: 
        UMG 3.3 has a lot of functionality that we don't need to use, including rebinnning.
        Our referencesuite comes with a rebinning module, so the rebinning/conversions should have already been done,
        and the group structure of the rebinned data should be the same as the rebinned structure.
        """
        assert np.shape(vector)==(self.n + 1,), "There must be exactly n+1={} group boundaries for n={} neutron groups.".format(self.n+1, self.n)\
            +"\nOnly continuous group structures are accepted."
        if self.verbosity>=2:
            print("Note that group structure will not affect unfolding output, so leaving group_structure as None will still give you the right answer.")
        self._group_structure = ary(vector)

    # Methods to write files
    def _write_phs_file(self, data_filename):
        """
        Writes the reaction rates measurements for UMG 3.3's multi-channel unfolding.
        a .phs file is used to provide the measurement used for multi-channel unfolding.
        The authors of UMG seems to regard cases where the measurements are continuous as mult-channel measurements. (e.g. proton-recoil spectra are continuous in energy.)
        (Of course we can discretize a measurement, or pretend discrete measurements are "continuous".
        This is what we will do here: pretend discrete measurements are continuous measurements.)

        The multi-channel UMG is written with the idea of converting a proton recoil spectrum back into a neutron spectrum.
        The proton spectrum will have MeV as its units on the x-axis (denoting the bounds of each bin),
            and "fluence rate per bin" as the y-axis.
        UMG assumes that, for a given neutron impulse (i.e. a beam of monoenergetic neutron),
            the proton-recoil response would be recorded as a histogram of pulse heights of equally spaced bins.
            The width of these bars is what the self.dummy_measurement_width refers to.
        And that the final proton recoil spectrum is the convolution of the measured neutron spectrum with their respective responses.
        However, in the context of neutron spectrum unfolding by foil activation, there is no such "proton spectrum" with "energy" on the x-axis and "fluence" on the y axis.
        Therefore where the UMG program asks for the relevant information, I have substituted MeV and "fluence rate per bin".
        dummy_measurement_width = 1eV is used as the width of the bin if a bin width is asked for.
        """
        with open(data_filename, 'w') as data_file:
            # Line 1: header
            data_file.write('Measurements obtained, file created by Python unfoldingsuite at '+str(datetime.now()).split('.')[0]+'\n')

            # Line 2 : MODE parameter and IEU parametr
            data_file.write(" {0: >11d} {1: >11d}\n".format(2, 1)) # forcing the unit of the "measurement" channel widths as MeV to make things simple

            # Line 3 : 
            data_file.write(" {0: >11d}".format(1))                           # dummy
            data_file.write(" {0: >11d} {0: >11d}".format(self.m))            # number of measurements (twice)
            data_file.write(" {0: >11.3e}\n".format(self.m*self.dummy_measurement_width/1E6))# highest measruement bin edge's energy

            # Line 4 : 
            for bin_number, (N_meas_i, sigma) in enumerate(zip(self.N_meas, self.sigma_N)):
                data_file.write("{0: >13.3e}".format((bin_number)*self.dummy_measurement_width/1E6))
                data_file.write("{0: >13.3e}".format(N_meas_i))
                data_file.write("{0: >13.3e}\n".format(sigma))
            data_file.write("{0: >13.3e}".format((self.m)*self.dummy_measurement_width/1E6))
            data_file.write("{0: >13.3e}".format(0))
            data_file.write("{0: >13.3e}\n".format(0))

    def _write_ibu_file(self, data_filename):
        """Write the reaction rates values to file for few channel (fc) UMG 3.3"""
        with open(data_filename, 'w') as data_file:
            # Line 1: 80 arb. characters (not read by UMG3.3)
            data_file.write('Measurements obtained, file created by Python unfoldingsuite at'+str(datetime.now()).split('.')[0]+'\n')

            # Line 2: Number of detectors - correction factor (0=no change to data)
            data_file.write(" {0: >4d} {1: >4d}\n".format(self.m, 0))

            # Line 3: Measured reaction rates
            for index, (N_meas_i, sigma) in enumerate(zip(self.N_meas, self.sigma_N)):
                data_file.write("RR{0: <6d}".format(index))             # The reaction label
                data_file.write("{0: >6.1f}".format(index))             # Bonner Sphere Diameter
                data_file.write("{0: >15.3e}".format(N_meas_i))         # The reaction rate
                data_file.write("{0: >15.3e}".format(sigma))            # The absolute uncertainty
                data_file.write("{0: >8.2f}".format(sigma/N_meas_i*100))  # % of uncertainty due to statistics
                data_file.write("{0: >8.2f}".format(0.0))               # % of uncertainty due to other sources
                data_file.write("{0: >6d}".format(1))                   # Flag - if negative, reaction ignored
                data_file.write("\n")

    def _write_rsp_file(self, data_filename):
        """
        Write reaction rates for multi channel (mc) UMG 3.3
        See _write_phs_file for more details.
        Note that the transpose of the response matrix has to written in this file,
            unlike _write_fmt_file where the respones matrix itself is written out.
        """
        with open(data_filename, 'w') as data_file:
            # LIne 1:  
            data_file.write('{0: >15e}\n'.format(self.dummy_measurement_width/1E6)) # it's just a dummy value denoting the "width" of the measurement channel. 
            # since foil-activation neutron spectrum unfolding doesn't have a "width" to the channel (there is either a gamma or no gamma decay from the foil).

            # Line 2:
            for (meanE, response) in zip(ary(self.group_structure[:-1] + np.diff(self.group_structure)/2)/1E6, self.Rm.T): # group_structure here must be provided in MeV; so that meanE is given with unit MeV
                # Line a:
                data_file.write('{0: >12e}'.format(meanE))                # energy of the incident particle
                data_file.write('{0: >6d}'.format(self.m))                # number of "measurement channels"
                data_file.write('{0: >13e}'.format(self.dummy_measurement_width/1E6*0))        # lower energy bin edge for the lowest "measurement channel"
                data_file.write('{0: >13e}\n'.format(self.dummy_measurement_width/1E6*self.m)) # upper energy bin edge for the lowest "measurement channel"
                # Line b: the reaction rate values
                data_file.write(_modified_array2string(response))

    def _write_fmt_file(self, data_filename):
        """Write response matrix to file for few channel (fc) UMG 3.3"""
        with open(data_filename, 'w') as data_file:
            # Line 1: 80 arb. characters
            data_file.write('Response matrix file created by Python unfoldingsuite at '+str(datetime.now()).split('.')[0]+'\n')

            # Line 2: 80 arb. characters
            data_file.write("Response matrix has shape ({},{})\n".format(self.m, self.n))

            # Line 3: number of energy bins = (channels + 1), and number indicating unit used.
            data_file.write(" {0: >9d} {1: >3d}\n".format(len(self.group_structure), 1)) # number indicating unit used: 0=eV, 1=MeV, 2=keV.

            # Line 4: Energy bin edges
            data_file.write(_modified_array2string(self.group_structure/1E6))

            # Line 5: Dummy value
            data_file.write(" {0: >9d}\n".format(0))

            # Line 6: The number of reaction rates
            data_file.write(" {0: >9d}\n".format(self.m))

            # Line 7: 
            for responseID, response in enumerate(self.Rm):
                # Line 7a: Detector reference
                data_file.write("RR{0: <6d}  RR{0: <14d}\n".format(responseID))     # ID + double space + long ID

                # Line 7b: Units and dummy values
                data_file.write("{0: >10.3e}  ".format(1.0))
                data_file.write("    cm^2")                           # 8 character response unit
                data_file.write(" {0: >9d} {0: >9d} {0: >4d} {0: >4d} {0: >4d} {0: >4d}\n".format(0)) # dummies

                # Line 7c: response valuse
                data_file.write(_modified_array2string(response))

    def _write_flu_file(self, data_filename):
        """Write a priori to file for both few channel and multi-channel UMG3.3"""
        with open(data_filename, 'w') as data_file:
            # Line 1: 80 arb. characters
            data_file.write("        {0}    *** ATTENTION: This file was specially compiled for UMG33\n".format(date.today()))

            # Line 2: MODE parameter and IEU (unit) parameter
            data_file.write(" {0: >11d} {1: >11d}\n".format(2, 1)) # the IEU parameter for the MEASUREMENT is chosen as 2 to avoid any scaling requirement.

            # Line 3:
            data_file.write(" {0: >11d}".format(1))                           # dummy
            data_file.write(" {0: >11d} {0: >11d}".format(self.n))            # number of bins (twice)
            data_file.write(" {0: >11.3e}\n".format(self.group_structure[-1]/1E6))# highest energy bin edge

            # Line 4: Lower energy bin edge, flux, flux uncertainty (only used by the IQU)
            for lower_bound, flux, flux_unc in zip(self.group_structure[:-1]/1E6, self.apriori, self.apriori_error):
                data_file.write("{0: >13.3e}".format(lower_bound))
                data_file.write("{0: >13.3e}".format(flux))
                data_file.write("{0: >13.3e}\n".format(flux_unc))
            data_file.write("{0: >13.3e}".format(self.group_structure[-1]/1E6))
            data_file.write("{0: >13.3e}".format(0))
            data_file.write("{0: >13.3e}\n".format(0))

    # Methods to read the solution files
    def _read_flu_file(self, UMG_outfilename, is_maxed_mc):
        """
        UMG programs output the flu file in flux per MeV (MODE parameter = 1),
        i.e. values are shown in (fluence rate per bin)/(width of the bin in E).
        Therefore we need to multiply by bin-width again to get the resulting fluence rate per bin.
        """
        try:
            with open(UMG_outfilename) as UMG_outfile:
                flux_per_energy = [float(line.strip('\n').split()[1]) for line in UMG_outfile.readlines()[3:3+self.n]] # skip first three line and last line (irrelevant data)
                if is_maxed_mc:
                    super()._append_phi(flux_per_energy*np.diff(self.group_structure/1E6)) # don't do any calculation on my side because I want this module to be purely an interface.
                else:
                    self.phi.append(flux_per_energy*np.diff(self.group_structure/1E6))
        except Exception as e:
            print("Unable to read data from the .flu file {} and convert it into the solution".format(UMG_outfilename))
            raise e

    def _read_txt_file(self, UMG_outfilename, is_maxed):
        """
        Read the chi2 per DoF from the .txt file;
        as well as the lambda vector outputted if MAXED is used        
        """
        with open(UMG_outfilename) as UMG_outfile:
            line = UMG_outfile.readline()
            while "final chi-squared p.d.f." not in line.lower(): #keep scrolling until we find this line.
                line=UMG_outfile.readline()

            # Using a separate attribute, self._txt_chi2_val, to record the chi^2 value as read from the txt file.
            # This is because it has a lower precision than the chi^2 value read from the stdout.
            try:
                self._txt_chi2_val = float(line.split()[-1])*self.m # this value has very low precision.
            except ValueError: # if it shows up as '******' instead, then we can just throw our hands up ¯\_(ツ)_/¯
                self._txt_chi2_val = line.split()[-1] # and say "This string '*****' is what we've got"

            if is_maxed: # continue scrolling to get the lambda vector as well if this is reading a maxed result
                while not ("DN" in line.upper() and 'LAMBDA' in line.upper()): # scroll until you reach the line with 'DN' and 'LAMBDA' in it.
                    line = UMG_outfile.readline()

                self.lambda_vector.append([]) # create an emtpy list to store the precise values
                while 'DN = detector number' not in line:
                    # read line
                    line = UMG_outfile.readline()

                    # convert if line isn't empty
                    if len(line.split())>0:
                        self.lambda_vector[-1].append( float(line.split()[-1]) ) # record the next element of the lambda vector
                    else:
                        break

                self.lambda_vector[-1] = ary(self.lambda_vector[-1]) # wrap it into the form of an ary

    def _read_maxed_fc_stdout(self, stdout):
        """
        Read the simulated annealing output
        """
        READING_LAMBDA = False
        lambda_vector = []
        IPRINT_EQ_1 = True # the maxed source code has a variable 'IPRINT' in it.
        # It controls how much information about the Simulated annealing will be printed
        # If the UMG MAXED executable is compiled with IPRINT!=1,
        # then I am not sure if the same algorithm listed below will work or not.
        for line in stdout.split('\n'):
            if IPRINT_EQ_1:
                if 'MAX FUNCTION VALUE SO FAR' in line: # This number can't be found anywhere else except in stdout.
                    self.umg_potential.append(Decimal(line.split()[-1]))
                if 'CHI-SQUARED PER DEGREE OF FREEDOM / MAXED' in line: # overwrite the number read from the txt file, since stdout gives it in a higher precision.
                    self.chi2_val.append(Decimal(line.split()[-1])*self.m)

                if READING_LAMBDA: # for recording the evolution of lambda, but it has a lower precision than self.vectors['lambda']
                    if len(line.split())>0:
                        for i in line.split():
                            lambda_vector.append(float(i))
                    else: 
                        READING_LAMBDA=False # stop reading lambda if we encounter an empty line.
                        self._append_lambda(ary(lambda_vector)) #and wrap up the current lambda vector.
                        lambda_vector = [] # create new lambda vector to be used in the next round.
                if line.strip()=='CURRENT OPTIMAL X' or line.strip()=='SOLUTION': #the line only contains the word(s) SOLUTION or CURRENT OPTIMAL X
                    READING_LAMBDA=True

            if 'IPRINT:' in line:
                if int(line.split('IPRINT:')[-1].split()[0])<1:
                    IPRINT_EQ_1 = False # if IPRINT<1 (i.e. I modified the Fortran code) then the loop above may not necessarily work (values may be missing).
            if self.verbosity>=3:
                print(line)

    def _read_stdout(self, stdout):
        """
        Read the stdout of all other algorithms. specifcally for GRAVEL.
        This has only been tested on GRAVEL_mc.
        """
        for line in stdout.split('\n'):
            if 'CHI-SQUARED' in line:
                try:
                    self.chi2_val.append(Decimal(line.split()[-1])*self.m)
                except InvalidOperation:
                    pass
            if self.verbosity>=3:
                print(line)

    def _read_plo_file(self, data_filename):
        """
        Read the phi variation across iterations.
        Only used for GRAVEL_mc and GRAVEL_fc
        """
        if self.verbosity>=2:
            print("reading plo file to extract the variation of phi across iterations for GRAVEL")

        with open(data_filename) as data_file:
            line = data_file.readline()
            while "1-th iteration" not in line:
                line = data_file.readline()
                if line=='':
                    return # we have scanned the whole file but no match
            three_vec = []
            while len(line)>0:
                line = data_file.readline()
                if len(line)==0: # if newline is empty: it must be EoF
                    break # escape loop, close file and return
                elif (not line[1].isnumeric()) or ('-th iteration' in line):
                    if len(three_vec)>0:
                        vector_extracted = [three_vec[2*(self.n+1)-1]]+three_vec[self.n+2:2*(self.n+1)-1] 
                        # for some reason the .plo file output has a three_vec[self.n+1:2*(self.n+1)-2] encodes all of the information required;
                        # but three_vec[2*(self.n+1)-1] repeats the first element.
                        # But on some cases the first element i.e.three_vec[(self.n+1)+1] gets zeroed for no reason.
                        phi_vec = ary(vector_extracted)*np.diff(self.group_structure)/1E6
                        self.phi.append(phi_vec)
                        self.N.append(self.Rm@phi_vec)
                        #chop up the three_vec 
                        three_vec = []
                else:
                    for i in line.split():
                        three_vec.append(float(i))
        return

    def _append_lambda(self, l_vec, step_size=1):
        self.lambda_vector.append(l_vec.copy())
        self.step_size.append(step_size)
        self.potential.append(self.potential_function(l_vec))
        super()._append_phi(self.lambda2spectrum(l_vec))
        self._potential_difference.append(self._potential_difference_function(l_vec))

    # Methods to write a collection of files, including control file.
    # Allow UMG to rescale the a priori spectra in all of them before unfolding.
    def _write_MAXED_fc_control_and_input_file(self, T=1, TR=.85):
        """
        Writes the control file and input files for few-channel MAXED.
        desired_chi2 is the desired output chi^2.
        """
        N_file, R_file, apfile = 'mxfc.ibu', 'mxfc.fmt', 'mxfc.flu'
        o_root = 'MAXED_fc'
        control_filename = 'mxfc.inp'
        self._write_ibu_file(N_file) # measurements (N_meas)
        self._write_fmt_file(R_file) # response matrix (Rm)
        self._write_flu_file(apfile) # a priori (apriori)
        with open(control_filename, 'w') as control_file:
            control_file.write(N_file.ljust(40)     +'file with data\n') # N_meas
            control_file.write(R_file.ljust(40)     +'file with response function\n') # Rm
            control_file.write(o_root.ljust(40)     +'name of output file\n')
            control_file.write(apfile.ljust(40)     +'file with default spectrum (DS)\n') # apriori
            
            control_file.write('{}'.format(self.group_structure[-1]/1E6*1.00001).ljust(40)                      +'maximum energy\n')
            control_file.write('{}'.format(self.desired_chi2/self.m).ljust(40)                                  +'chi-square factor\n')
            control_file.write('{},{}'.format(T, TR).ljust(40)                                                  +'initial temperature, temperautre reduction factor\n')
            control_file.write('{},{}'.format(2,2).ljust(40)                                                    +'2=Use group structure of DS, 1=plot as d(phi)/dE \n')
            control_file.write('{}'.format(1).ljust(40)                                                         +'Use a scale factor for the DS ?(1=True)\n')
            control_file.write('{}'.format(0).ljust(40)                                                         +'0 = scale factor chosen by UMG\n')
        return N_file, R_file, apfile, control_filename, o_root

    def _write_MAXED_mc_control_and_input_file(self):
        """
        Writes the control file and input files for multi-channel GRAVEL.
        """
        N_file, R_file, apfile = 'mxmc.phs', 'mxmc.rsp', 'mxmc.flu'
        o_root = 'MAXED_mc'
        control_filename = 'mxmc.inp'
        self._write_phs_file(N_file) # measurements (N_meas)
        self._write_rsp_file(R_file) # response matrix (Rm)
        self._write_flu_file(apfile) # a priori (apriori)
        if self.desired_chi2==0:
            if self.verbosity>=1:
                print("Warning: using GRAVEL may only asymptotically approach desired_chi2 = 0")
        with open(control_filename, 'w') as control_file:
            control_file.write(N_file.ljust(40)     +'file with data\n') # N_meas
            control_file.write(R_file.ljust(40)     +'file with response function\n') # Rm
            control_file.write(o_root.ljust(40)     +'name of output file\n')
            control_file.write(apfile.ljust(40)     +'file with default spectrum (DS)\n') # apriori
            
            # NEED TO TEST IF -1 and +1 is necessary/ works in the rest of the functions too
            control_file.write('{},{}'.format(0, self.dummy_measurement_width*self.m/1E6).ljust(40)            +'lo, hi MC E bin edges (in MC E units)\n')
            control_file.write('{},{}'.format(self.group_structure[0]/1E6, self.group_structure[-1]/1E6).ljust(40) +'lo, hi RF E bin edges (in MeV)\n')
            control_file.write('{}'.format(self.desired_chi2/self.m).ljust(40)                                  +'chi-square factor\n')
            control_file.write('{}'.format(self.max_iter).ljust(40)                                             +'Maximum number of iter. in L-BFGS-B\n')
            control_file.write('{},{}'.format(2,2).ljust(40)                                                    +'2=Use group structure of DS, 2=plot as Ed(phi)/dE \n') # seems to fail when pltting as d(phi)/dE instead. Not sure why.
            control_file.write('{}'.format(1).ljust(40)                                                         +'Use a scale factor for the DS ?(1=True)\n')
            control_file.write('{}'.format(0).ljust(40)                                                         +'0 = scale factor chosen by UMG\n')
        return N_file, R_file, apfile, control_filename, o_root

    def _write_GRAVEL_fc_control_and_input_file(self, plot_interval=1):
        """
        Writes the control file and input files for few-channel GRAVEL.
        """
        N_file, R_file, apfile = 'gvfc.ibu', 'gvfc.fmt', 'gvfc.flu'
        o_root = 'GRAVELfc'
        control_filename = 'gvfc.inp'
        self._write_ibu_file(N_file) # measurements (N_meas)
        self._write_fmt_file(R_file) # response matrix (Rm)
        self._write_flu_file(apfile) # a priori (apriori)
        with open(control_filename, 'w') as control_file:
            control_file.write(N_file.ljust(40)     +'file with data\n') # N_meas
            control_file.write(R_file.ljust(40)     +'file with response function\n') # Rm
            control_file.write(o_root.ljust(40)     +'name of output file\n')
            control_file.write(apfile.ljust(40)     +'file with default spectrum (DS)\n') # apriori

            control_file.write('{}'.format(self.group_structure[-1]/1E6*1.00001).ljust(40)                      +'maximum energy\n')
            control_file.write('{}'.format(self.desired_chi2/self.m).ljust(40)                                  +'chi-square factor\n')
            control_file.write('{},{}'.format(self.max_iter, plot_interval).ljust(40)                           +'max # iterations, frequency of plotting\n')
            control_file.write('{},{}'.format(2,2).ljust(40)                                                    +'2=Use group structure of DS, 2=plot as Ed(phi)/dE \n') # seems to fail when pltting as d(phi)/dE instead. Not sure why.
            control_file.write('{}'.format(1).ljust(40)                                                         +'Use a scale factor for the DS ?(1=True)\n')
            control_file.write('{}'.format(0).ljust(40)                                                         +'0 = scale factor chosen by UMG\n')
        return N_file, R_file, apfile, control_filename, o_root

    def _write_GRAVEL_mc_control_and_input_file(self, plot_interval=1):
        """
        Writes the control file and input files for mutli-channel GRAVEL.
        desired_chi2 is the desired output chi^2.
        """
        N_file, R_file, apfile = 'gvmc.phs', 'gvmc.rsp', 'gvmc.flu'
        o_root = 'GRAVELmc'
        control_filename = 'gvmc.inp'
        self._write_phs_file(N_file) # measurements (N_meas)
        self._write_rsp_file(R_file) # response matrix (Rm)
        self._write_flu_file(apfile) # a priori (apriori)
        if self.desired_chi2==0:
            if self.verbosity>=1:
                print("Warning: using GRAVEL may only asymptotically approach desired_chi2 = 0")
        with open(control_filename, 'w') as control_file:
            control_file.write(N_file.ljust(40)     +'file with data\n') # N_meas
            control_file.write(R_file.ljust(40)     +'file with response function\n') # Rm
            control_file.write(o_root.ljust(40)     +'name of output file\n')
            control_file.write(apfile.ljust(40)     +'file with default spectrum (DS)\n') # apriori
            
            # NEED TO TEST IF -1 and +1 is necessary/ works in the rest of the functions too
            control_file.write('{},{}'.format(0, self.dummy_measurement_width*self.m/1E6).ljust(40)            +'lo, hi MC E bin edges (in MC E units)\n')
            control_file.write('{},{}'.format(self.group_structure[0]/1E6, self.group_structure[-1]/1E6).ljust(40) +'lo, hi RF E bin edges (in MeV)\n')
            control_file.write('{}'.format(self.desired_chi2/self.m).ljust(40)                                  +'chi-square factor\n')
            control_file.write('{}'.format(self.max_iter, plot_interval).ljust(40)                              +'Maximum number of iter. in L-BFGS-B\n')
            control_file.write('{},{}'.format(2,2).ljust(40)                                                    +'2=Use group structure of DS, 2=plot as Ed(phi)/dE \n') # seems to fail when pltting as d(phi)/dE instead. Not sure why.
            control_file.write('{}'.format(1).ljust(40)                                                         +'Use a scale factor for the DS ?(1=True)\n')
            control_file.write('{}'.format(0).ljust(40)                                                         +'0 = scale factor chosen by UMG\n')
        return N_file, R_file, apfile, control_filename, o_root

    def run(self, method, executable_path, delete_files=True, catch_error=False, **kwargs):
        """
        Creates files locally;
        Calls executable from executable_path to operate on these local files;
        Read the results.
        
        Note that it will overwrite any existing file from the previous runs.

        call signature example:
        self.run('MAXED_mc', '../maxed_mc.exe', delete_files=True, catch_error=True)
        available methods = 
        """
        if ('GRAVEL' in method) and self.desired_chi2 == 0:
            if self.verbosity>=1:
                print("GRAVEL is an iterative solver with no means of achieving exactly desired_chi2=0.")
                print("This means GRAVEL may run until self.max_iter is reached.")
        files = getattr(self,'_write_'+method+'_control_and_input_file')(**kwargs)
        data_files, control_file, out_file_root = files[:-2], files[-2], files[-1]
        assert os.path.exists(executable_path), "Please point to a valid executable file for "+method
        p = subprocess.run([os.path.abspath(executable_path), control_file], capture_output=True)

        # check that it has been properly executed.
        if p.returncode==0:
            self.stdout = p.stdout.decode()
        else:
            error_msg = method+' failed with return code '+str(p.returncode)
            if catch_error:
                print(error_msg)
                return p # stop here.
            else:
                raise RuntimeError(error_msg)

        # read the results.
        if method.startswith('MAXED'):
            self.solution = SolutionExtractor_UMG_MAXED(self)
            self.lambda_vector = []
        if method == 'MAXED_fc':
            # records a lists of scalars
            self.potential = []
            self.umg_potential = [] 
            self._potential_difference = []
            # and clear out the phi and chi2_val (because we have started with fDEF = re-scaled apriori instead of just apriori)

            # calculate the best fit yield for apriori.
            # This yield can also be found in the txt output file on the line with "Scaling Factor/Default Spectrum"
            # or the stdout on the line with "SCALE FACTOR/DEFAULT SPEC. FOR BEST FIT"
            N_ap = self.Rm @ self.apriori
            self._apriori_scale_factor = (self.N_meas @ self.covar_N_inv @ N_ap)/(N_ap @ self.covar_N_inv @ N_ap) # rescale apriori so that the chi2 is at its minimum
            self.fDEF = self._apriori_scale_factor * ary(self.apriori)

            # a further complication is that UMG scales everything so that sum(fDEF) = 1.
            self._fDEF_scaled = self.fDEF/self.fDEF.sum()
            self._N_meas_scaled = self.N_meas/self.fDEF.sum()
            self._sigma_N_scaled = self.sigma_N/self.fDEF.sum()
            self._covar_N_scaled = self.covar_N/self.fDEF.sum()**2
            
            # record a list of vectors
            self._read_maxed_fc_stdout(self.stdout) # reads the lambda_vectors and the final chi2_val.
        else:
            self._read_stdout(self.stdout) # reads all of the chi2_val for GRAVEL_fc, GRAVEL_mc, MAXED_mc.

        if method.startswith('GRAVEL'):
            self._read_plo_file(method.replace('_','')+'.plo')

        self._read_flu_file(out_file_root+'.flu', method=='MAXED_mc')

        # fool-proof way of reading in the final chi2_val (and final lambda_vector in case of MAXED's)
        self._read_txt_file(out_file_root+'.txt', method.startswith('MAXED')) # Therefore this must be used as the last line in the reading block.

        # delete the files generated.
        if delete_files:
            [os.remove(data_file) for data_file in data_files] # remove the input files
            os.remove(control_file) # remove the control file
            # remove the output files
            [os.remove(out_file_root+extension) for extension in ['.txt', '.flu', '.plo']]
            if method.endswith('mc'):
                os.remove(out_file_root+'.gru')
            if method.startswith('MAXED'):
                os.remove(out_file_root+'.par')
            if method=='MAXED_mc':
                os.remove('iterate.dat')

    run.__doc__+=str(available_methods)

    def _newlambda2old(self, long_lambda):
        """Convert the new format lambda vector into old format lambda vector"""
        l_vec = long_lambda[:-1] + long_lambda[-1]/ary(self._sigma_N_scaled)
        return l_vec

    def potential_function(self, l_vec):
        """Calculate the potential value for the the new format lambda vector"""
        if len(l_vec)==self.m+1:
            l_vec = self._newlambda2old(l_vec)
        exponent = (l_vec) @ self.Rm
        exponent = np.clip(exponent, -180, 174) # clip it to avoid float error after exponentiation
        term1 = (self._fDEF_scaled) @ exp(-exponent) # maximize the flux
        term3 = sqrt(self.desired_chi2 * (l_vec @ self._covar_N_scaled @ l_vec)) # ensure the solution.N matches desired chi2
        term4 = l_vec @ self._N_meas_scaled # minimize the lambda vector
        Z = -term1 -term3 -term4 # all three terms are scaled by exactly the same factor of self.fDEF.sum()
        # (assuming the loss function is implemetned correctly; which UMG isn't) so there shouldn't be any distortion.

        if self.verbosity>=3:
            print('-term1, -term2, -term3 =', -term1, -term3, -term4)
            print(Z)
            print('----------------------------------')
        return Z

    def _potential_difference_function(self, long_lambda):
        """
        Calculates the difference between the correct and incorrect loss functions.
        umg_potential = potential + potential_difference
        """
        l_vec = self._newlambda2old(long_lambda)
        correct = sqrt(self.desired_chi2 * (l_vec @ self._covar_N_scaled @ l_vec)) 
        incorrect = sqrt(self.desired_chi2 * (long_lambda[:-1]@ self._covar_N_scaled @ long_lambda[:-1]))
        return correct - incorrect

    def lambda2spectrum(self, l_vec):
        if len(l_vec)==self.m+1:
            l_vec = self._newlambda2old(l_vec)
        exponent = l_vec @ self.Rm
        exponent = np.clip(exponent, -180, 174)
        final_flux = self.fDEF * exp(-exponent)
        return final_flux

    def get_covar_phi_inv(self, l_vec, alt_def=False):
        """
        Try to get the inverse of the covariance matrix of the output spectrum.
        However I think I've made a mistake on both of them.
        """
        phi = self.lambda2spectrum(l_vec)
        if alt_def:
            Rm_inv = pinv(self.Rm)
            denominator = l_vec @ self.covar_N @ l_vec
            S_phi = Rm_inv @ self.covar_N @ Rm_inv.T

            curvature_of_phi = np.diag(np.ones(self.n) - (Rm_inv@self.N_meas)/phi**2) 
            curvature_of_phi -= sqrt(self.desired_chi2/denominator**3) * (S_phi @ ln(phi/self.fDEF)) * self.fDEF/phi
            curvature_of_phi += sqrt(self.desired_chi2/denominator) * S_phi @ np.diag(self.fDEF*(1-ln(phi/self.fDEF))/phi**2)
        else:
            modifier = 1/phi * self.Rm
            covar_lambda_inv = self.Hessian(l_vec)
            curvature_of_phi = (modifier.T) @ covar_lambda_inv @ modifier
        return - curvature_of_phi

    def _print_method(self):
        print_UMG3_3_INTERFACE()

    @classmethod
    def copy_from(cls, unf, *args, **kwargs):
        """
        Similar to the default copy_from in datahandler.py; but without any produced 
        """
        if not issubclass(type(unf), UnfoldingDataLoader):
            raise TypeError("This method is used for copying info from one instance into another.")

        new_kwargs = {attr: getattr(unf, attr) for attr in cls._essential_attr_list if hasattr(unf, '_'+attr)}

        if hasattr(unf, 'group_structure'):
            new_kwargs.update({'group_structure': unf.group_structure})
        # verbosity value copied into the new instance.
        new_kwargs.update({'verbosity': unf.verbosity})

        # manual inputs overrides the rest
        new_kwargs.update(kwargs)

        return cls(**new_kwargs)


# Need to re-write this to make sure that I extract sufficient information from each of the four algorithms.
# Make sure to test both mc cases as well.
# For MAXED_fc I don't need to test more than IPRING>1.