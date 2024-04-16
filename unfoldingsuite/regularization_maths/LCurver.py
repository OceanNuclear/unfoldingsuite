import matplotlib.pyplot as plt
import pandas as pd
import numpy as np; from numpy import log as ln
from numpy.linalg import inv, det
from numpy import diag, sqrt, exp
import time
from scipy.stats import gmean, linregress, describe

from correctedinversemethod import BareRegularizer
from types import MethodType
from scipy.optimize import fsolve, newton
import seaborn as sns
import json
from collections import OrderedDict

JOYPLOT = True
if JOYPLOT: 
    NUM_SAMPLE = 10
    PRETTY= True
else:
    NUM_SAMPLE = 50
TAU=1E-9
FSOLVE_APPROACH = False
easy_logspace = lambda start, end, num: np.logspace(*np.log10([start, end]), num)

def cross_over_search_log_space( func, start, stop, depth=0, num_slice=4, max_depth=12): #recursive
    '''
    plug in a function f(x) that returns a boolean value that indicates whether the cross-over point has occurred or not.

    '''
    range_to_search = easy_logspace(start, stop, num_slice)
    range_to_search = [float(i) for i in range_to_search]
    arg_vec = np.argwhere(np.diff([ func(i) for i in range_to_search ]))
    assert len(arg_vec)==1, f"expected one and only one cross-over point in this range, instead we have {len(arg_vec)}."
    crossing = arg_vec[0][0]
    if depth<max_depth:
        print()
        return cross_over_search_log_space(func, range_to_search[crossing], range_to_search[crossing+1], depth+1, num_slice, max_depth)
    else:
        print()
        xmin = range_to_search[crossing]
        xmax = range_to_search[crossing+1]
        if func(xmin): #Stay on the small side of the small side evaluates to True
            print()
            return xmin
        else: #Stay on the large side of the large side evaluates to True
            print()
            return xmax

if __name__=="__main__":
    num_tol = np.finfo(TAU).max #largest number that the computer can store safely
    apriori = pd.read_csv('a_priori.csv', header=None)
    answer  = pd.read_csv('answer_spectrum.csv',header=None).values.reshape([-1])
    #load in the group structure
    with open('tbmd175.json', 'r') as f:
        gs = json.load(f)['group structure']
    #load in the response matrix
    R       = pd.read_csv('R.csv', header=None, index_col=0)
    #reaction rates
    rr      = pd.read_csv('reaction_rates.csv', header=None)
    sigma   = sqrt(rr.values.reshape([-1]))

def geometric_percentile(lower, upper, perc=50):
    diff = ln(upper) - ln(lower)
    return exp(ln(lower) + diff*perc/100)

def conv2PUL(gs, flux_integrated):
    leth_space = np.diff(ln(gs))
    return flux_integrated/leth_space

def extrap_to_zero(x1, y1, x2, y2):
    x_diff = x2-x1
    y_diff = y2-y1
    return x1 - y1 * x_diff/y_diff

def lingress(only_y):
    return linregress(np.arange(len(only_y)), only_y)

class SmartRegularizer(BareRegularizer):
    def __init__(self, *args, tau_min_max=None, safety_margins=None, **kwargs):
        super().__init__(*args, **kwargs)
        if tau_min_max is None:
            self.tau_min, self.tau_max = self.valid_tau_range()
        else:
            self.tau_min, self.tau_max = tau_min_max
        self.safety_margins = safety_margins

    def copy(self): # overwrite the existing method
        return SmartRegularizer(self.N, self.S_N, self.R, self.fDEF, tau_min_max=[self.tau_min, self.tau_max], safety_margins=self.safety_margins)
        
    def valid_tau_range(self):
        '''
        Finds the range of tau where the determinant is finite.
        '''

        np.seterr('ignore') # allow overflows without failing
        test_range = np.linspace(-25, 25, num=201)
        range_of_tau_exp, range_d = [], []
        for t in test_range:
            d = self.log2_detH(2**t)
            if np.isfinite(d):
                range_of_tau_exp.append(t)
                range_d.append(d)
        #Find the lower and upper bounds for detH to exist in
        minexp = np.finfo(2**range_of_tau_exp[ 0]).minexp
        maxexp = np.finfo(2**range_of_tau_exp[-1]).maxexp
        if FSOLVE_APPROACH:
            # To find it more exactly, use fsolve:
            # function created for fsolve to find roots for:
            lower_detH = lambda t: self.log2_detH(2.0**t) - minexp #calculate how close it is to the smallest computable number in log2 space
            upper_detH = lambda t: self.log2_detH(2.0**t) + maxexp #calculate how close it is to the largest computable number in log2 space
            # lower_detH = lambda t: self.log2_detH(2.0**t) > minexp #calculate whether it has exceeded the float limitation or not
            # upper_detH = lambda t: self.log2_detH(2.0**t) < maxexp #calculate whether it has exceeded the float limitation or not
            #using a root finder imported from scipy,
            min_exp = fsolve(lower_detH, min(range_of_tau_exp)) # solve for the minimum, using the minimum in the allowed range of tau as the starting point
            max_exp = fsolve(upper_detH, max(range_of_tau_exp)) # solve for the maximum, using the maximum in the allowed range of tau as the starting point
            np.seterr('print') # unset ignoring the error

            return float(2**min_exp), float(2**max_exp)
        else:
            #Taking advantage of the fact that detH varies as exp(tau), use linear extrapolation in log2(detH) space.
            lower_estimates, upper_estimates = [], []
            for i in range(min([4, len(range_of_tau_exp)])):
                lower_estimates.append(extrap_to_zero(range_of_tau_exp[ i], range_d[ i]-minexp, range_of_tau_exp[ i+1], range_d[ i+1]-minexp))
                upper_estimates.append(extrap_to_zero(range_of_tau_exp[-i], range_d[-i]-maxexp, range_of_tau_exp[-i-1], range_d[-i-1]-maxexp))
            return 2**max(lower_estimates), 2**min(upper_estimates) # use max(lower), min(upper) to be safe
        #Other, more accurate approaches to determine the uppwer and lower bound of usable taus include:
        # using np.linalg.cond(test_matrix)to determine the lower limit of tau (increases as tau approaches 0)
        # using np.linalg.eigvals(test_matrix) or np.linalg.eigvals(test_matrix) to determine the upper bounds (highest number increases up to 1E308 before breaking inv to give )
        
    def log2_detH(self, t):
        # get the log2 value of the determinant of the invertible matrix.
        f = self.fDEF
        if len(self.f)>0:
            f = self.f[-1]
        right_vec = self.S_N @ self.R @ f
        Y = (self.N @ right_vec)/( (self.R @ f) @ right_vec)
        hessian_matrix = ( Y* self.R.T ) @ self.S_N @ (Y * self.R) + t * np.diag(self.fDEF/(f**2))
        return np.log2( det(hessian_matrix) )

    def run_to_convergence(self, speed_up_at=5000, verbose=True, strict=True):
        starttime = time.time()
        def print_termination_statement():
            if verbose:
                print("Time taken to converge = {:>6.2f}s for tau={:>9.9e} within {:>5d} iterations at chi^2={:>9.9e}".format(
                    time.time()-starttime, self.tau, len(self.f), self.chi2_val[-1]))
        
        # while True:
        for it in range(5000):
            self.take_step()
            if all([wi==1.0 for wi in self.w[-20:]]) and it > 20:# last 20 steps all had w==1.0
                if strict:
                    if self.termination_cond(['Y','steepness','chi2_val', 'loss_val', 'reg_val'], prev_iters=20):
                        print_termination_statement()
                        return
                else:
                    print_termination_statement()
                    return
        self.alpha = 0.85
        print(f"Convergence too slow, switching to higher value of alpha={self.alpha}")
        while True:
            self.take_step()
            if all([wi==1.0 for wi in self.w[-20:]]):# last 20 steps all had w==1.0
                if strict:
                    if self.termination_cond(['Y','steepness','chi2_val', 'loss_val', 'reg_val'], prev_iters=20):
                        print_termination_statement()
                        return
                else:
                    print_termination_statement()
                    return

    def clean_rerun(self, t, **kwargs):
        new_reg = self.copy()
        new_reg.tau = t
        new_reg.run_to_convergence(**kwargs)
        # chi2_val = new_reg.chi2_val[-1]
        # reg_val = new_reg.reg_val[-1]
        return new_reg

    def get_safety_margins(self):
        print("Trying to calculate the reasonable range of chi2 and regularization value.")
        
        new_reg = self.copy()
        new_reg.tau = float(new_reg.tau_min)
        new_reg.run_to_convergence(strict=False)
        chi2_min, reg_max = new_reg.chi2_val[-1], new_reg.reg_val[-1]

        new_reg = self.copy()
        new_reg.tau = float(new_reg.tau_max)
        new_reg.run_to_convergence(strict=False)
        chi2_max, reg_min = new_reg.chi2_val[-1], new_reg.reg_val[-1]

        self.safety_margins = OrderedDict({'chi^2/2':(chi2_min, chi2_max),
                                            'D_KL':(reg_min, reg_max)})

    def default_run(self, percentile=50, **kwargs):
        t = geometric_percentile(self.tau_min, self.tau_max, percentile)
        self.tau = t
        self.run_to_convergence(**kwargs)

    def run(self, method='run_chi2_per_dof_eq_1', *args, **kwargs):
        '''
        default method is adjust_tau_to_chi2_eq (equal self.m)
        Other run methods include:
        default_run # geometric mean of the allowed bounds of the tau value
        run_to_convergence # 1E-7 by default
        adjust_tau_to_reg_val_eq
        '''
        if 'adjust' in method:
            t = getattr(self, method)(*args, **kwargs)
            self.tau = t
            self.run_to_convergence()
        else:
            return getattr(self, method)(*args, **kwargs)

    def adjust_tau_to_chi2_eq(self, chi2_val_target, strict=True, verbose=False):
        if self.safety_margins is None:
            self.get_safety_margins()
        (chi2_min, chi2_max), _ = self.safety_margins.values()
        
        gm = gmean([self.tau_min, self.tau_max])
        obj_func = lambda t: (self.clean_rerun(float(t), strict=strict, verbose=verbose).chi2_val[-1]-chi2_val_target)>0
        if (chi2_min<=chi2_val_target<=chi2_max):
            log_obj_func = lambda t: obj_func(exp(t))
            return exp(newton(log_obj_func, x0=ln(gm), x1=ln(self.tau_min))) #, tol=1E-12
            # return newton(obj_func, x0=gm, x1=self.tau_min)
        else:
            if chi2_val_target<chi2_min: # need more freedown, less restraint
                direction = -1
                current_chi2_val = chi2_min
                test_tau = self.tau_min
            elif chi2_val_target>chi2_max: # need more lasso-ing
                direction = +1
                current_chi2_val = chi2_max
                test_tau = self.tau_max
            else:
                raise ValueError(f'Invalid input to {chi2_val_target=} detected.')
            print(f"No guarantee of convergence when using {chi2_val_target=} outside of this range {chi2_min=}, {chi2_max=}.\nDeviating from this safe range to begin search...")
            chi2_val_records = [current_chi2_val,]
            taus = [test_tau,]
            # keep scaling the tau until it overshoots the reg_val_target
            while np.sign(chi2_val_target - current_chi2_val)==direction:
                test_tau *= 10**(direction)
                current_chi2_val = self.clean_rerun(test_tau, strict=strict, verbose=verbose).chi2_val[-1]
                taus.append(test_tau)
                chi2_val_records.append(current_chi2_val)
            print(f'the value of tau that give the required chi2_val_target is between', taus[-2:], 'giving chi2 val=', chi2_val_records[-2:])
            t = cross_over_search_log_space(obj_func, *taus[-2:])
            print('The corresponding value of tau is found as', t)
            return t

    def adjust_tau_to_reg_val_eq(self, reg_val_target, strict=True, verbose=False):
        if self.safety_margins is None:
            self.get_safety_margins()
        _, (reg_min, reg_max) = self.get_safety_margins()
        
        gm = gmean([self.tau_max, self.tau_min])
        obj_func = lambda t: (self.clean_rerun(float(t), strict=strict, verbose=verbose).reg_val[-1]-reg_val_target)>0
        if (reg_min <= reg_val_target<=reg_max):
            log_obj_func = lambda t: obj_func(exp(t))
            return exp(newton(log_obj_func, x0=ln(gm), x1=ln(self.tau_max))) #, tol=1E-12
            # return newton(obj_func, x0=gm, x1=self.tau_max)
        else:
            if reg_val_target<reg_min: # need more lasso-ing
                direction = +1
                current_reg_val = reg_min
                test_tau = self.tau_max
            elif reg_val_target>reg_max: # need less restraint
                direction = -1
                current_reg_val = reg_max
                test_tau = self.tau_min
            else:
                raise ValueError(f'Invalid input to {reg_val_target=} detected.')
            print(f"No guarantee of convergence when using {reg_val_target=} outside of this range {reg_min=}, {reg_max=}.\nDeviating from this safe range to begin search...")
            reg_val_records = [current_reg_val,]
            taus = [test_tau,]
            # keep scaling the tau until it overshoots the reg_val_target
            while np.sign(chi2_val_target - current_reg_val)==direction:
                test_tau *= 10**(direction)
                current_reg_val = self.clean_rerun(test_tau, strict=strict, verbose=verbose).reg_val[-1]
                taus.append(test_tau)
                reg_val_records.append(current_reg_val)
            print(f'the value of tau that give the required reg_val_target is between', taus[-2:], 'giving reg val=', reg_val_records[-2:])
            t = cross_over_search_log_space(obj_func, *taus[-2:])
            print('The corresponding value of tau is found as', t)
            return t

    def termination_cond(self, list_of_attr, prev_iters=20, rvalue_abs_threshold=0.35, fractional_threshold=2.5E-2):
        '''
        The optimal paramter must be those that can reduce the . 
        To keep the computational requirement on lingress low, I've chosen prev_iters=20.
        At prev_iters=20, for abs(rvalue) can sometimes go up to 0.16.. Therefore I have chosen to keep the rvalue_abs_threshold>0.25.
        And when 'steepness' is included in the list_of_attr, the fractional_threshold has to be strictly higher than 2%.
        Meanwhile, the slope can approach zero when convergence is reached (as long as any cyclic variation's period is shorter than prev_iters/2).
        '''
        bool_list = []
        for attr in list_of_attr:
            trend = lingress(getattr(self, attr)[-prev_iters:])
            this_quantity_should_terminate = True
            if fractional_threshold: # if fractional_threshold !=False and !=None
                if abs(trend.slope/(getattr(self, attr)[-1])) > fractional_threshold:
                    this_quantity_should_terminate = False
            if abs(trend.rvalue) > rvalue_abs_threshold:
                this_quantity_should_terminate = False
            bool_list.append(this_quantity_should_terminate)
        return all(bool_list)

    def run_chi2_per_dof_eq_1(self, chi2_per_dof=1, strict=True, **kwargs):
        tau_that_gives_chi2_dof_eq_1 = self.adjust_tau_to_chi2_eq( chi2_per_dof * self.m , strict=strict, **kwargs)
        self.tau = tau_that_gives_chi2_dof_eq_1
        self.run_to_convergence()

if __name__=="__main__":
    reg = SmartRegularizer(rr.values.reshape([-1]), np.diag(1/sigma**2), R.values, apriori.values.reshape([-1]))
    trange = easy_logspace(*reg.valid_tau_range(), NUM_SAMPLE)
    trange = [float(i) for i in trange]
    chi2_val_list, reg_val_list, iterations = [], [], []
    f = []
    for t in trange:
        reg = reg.clean_rerun(t, verbose=False)
        chi2_val, reg_val = reg.chi2_val[-1], reg.reg_val[-1]
        #for some reason reg is not being updated globally...
        chi2_val_list.append(chi2_val)
        reg_val_list.append(reg_val)
        #once outside of clean_rerun, the t gets reset back to 1E-7 for some reason.
        iterations.append(len(reg.f))
        f.append(conv2PUL( gs, reg.f[-1])) #PUL looks better
        # f.append(reg.f[-1])

    '''
    starttime = time.time()
    chi2_val, reg_val, iterations = [], [], []
    f = []
    # trange = np.linspace(*valid_tau_range(reg), NUM_SAMPLE), 
    trange = easy_logspace(*valid_tau_range(reg), NUM_SAMPLE)

    reso = np.finfo(reg.tau).resolution
    for t in trange:
        reg = reg.copy()
        reg.tau = t
        while True:
            reg.take_step()
            if all(reg.steepness[-2:] < 5* reso):
                break #break when the last two iterations has a smaller than the possible resolution of the program
        chi2_val.append(reg.chi2_val[-1])
        reg_val.append(reg.reg_val[-1])
        iterations.append(len(reg.f))
        f.append(reg.f[-1])
        print(f"time passed = {time.time() - starttime} s")
    '''
    if True:
        x_loc = np.array([(ln(gs[j])+ln(gs[j+1]))/2 for j in range(len(reg.f[-1]))]) #The accurate representation
        def transformE(GS):
            new_arr = []
            for g in GS:
                if 7.5>g:
                    new_arr.append((-g+7.5)*0.9 + g)
                elif 7.5<=g<14.5:
                    new_arr.append((g-7.5)/7+7.5)
                elif g>14.5:
                    new_arr.append(g-14.5 + 8.5)
            return np.array(new_arr)
        x_loc = transformE(x_loc)
    if JOYPLOT:
        # df = [ [(ln(f[i][j]), trange[i], x_loc[j]) for j in range(len(f[i]))] for i in range(len(trange))]
        df = [ [(ln(f[NUM_SAMPLE-i-1][j]), trange[i], x_loc[j]) for j in range(len(f[i]))] for i in range(len(trange))] # reversing the order
        df = np.array(df).reshape([-1, 3])
        df = pd.DataFrame(df, columns=['f', 'tau', 'gs_logged'])
        sns.set(style="white", rc={"axes.facecolor": (.27,.27,.27,.0)})
        #old palette
        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)[::-1]
        #new palette
        # pal = sns.color_palette('coolwarm', NUM_SAMPLE)

        cmap = plt.get_cmap('viridis')
        # cmap = plt.get_cmap('plasma')
        # cmap = plt.get_cmap() # default jet colormap
        # cmap = plt.get_cmap('twilight_shifted')
        # pal = [tuple(c)[::-1] for num, c in enumerate(cmap.colors[::]) if num//NUM_SAMPLE==num/NUM_SAMPLE]
        g = sns.FacetGrid(df, row='tau', hue='tau', aspect = 15, height=1, palette=pal)

        g.map(plt.fill_between, 'gs_logged', 'f', y2=df['f'].min(), clip_on=False, alpha=1, lw=1.5)
        g.map(plt.axhline, y=df['f'].min(), clip_on=False)
        g.fig.subplots_adjust(hspace=-.82)
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .5, r"$\tau=$"+label, ha='left', va='center', transform=ax.transAxes)
        # g.map(label,'tau')
        #Color scheme: faster colour change, more obvious variation
        g.despine(bottom=True, left=True)
        g.set_titles('') # don't put any titles
        # plt.title(r"Variation of unfolded neutron spectrum wrt. $\tau$")
        g.set(yticks=[])
        g.set(xlim=(min(x_loc), max(x_loc)))
        # plt.xlabel('bin number')
        plt.xticks([])
        # plt.show()
        plt.savefig("./Results/New/JoyPlot.png", facecolor='#909090', dpi=300, figsize=(15,10))
        plt.cla()
    else:
        plt.plot(chi2_val_list, reg_val_list, marker='x')
        plt.xlabel(r"$\chi^2$")
        plt.ylabel(r"Relative entropy ($D_{KL}$)")
        plt.savefig("./Results/New/L_curve.png")
        plt.cla()
        plt.plot(trange, np.array(iterations)-1, marker='x')
        plt.title("Speed of convergence")
        plt.xlabel(r"$\tau$")
        plt.ylabel("Number of iterations required to converge onto a stable solution")
        plt.savefig("./Results/New/NumIter.png", facecolor='')

# if __name__=='__main__':
if False:
    reg = SmartRegularizer(rr.values.reshape([-1]), np.diag(1/sigma**2), R.values, apriori.values.reshape([-1]))
    t = reg.adjust_tau_to_chi2_eq(3.97E-06)
    reg = reg.copy()
    reg.tau = float(t)
    reg.run_to_convergence()
