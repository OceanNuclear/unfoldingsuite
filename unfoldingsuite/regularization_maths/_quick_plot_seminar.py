from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
tau = 2*pi
import numpy as np;
from matplotlib import pyplot as plt
import pandas as pd
import sys
import seaborn as sns
import json

with open('../reg_benchmark/SelectedSpectra.json') as f:
    specs = json.load(f)

specs = pd.DataFrame(specs).T

R_df = pd.read_csv('R.csv', header=None, index_col=[0])

with open('tbmd175.json') as f:
    gs = json.load(f)['group structure']

def step_plot(y, *args, **kwargs):
    return plt.step(gs, (list(y)+[y[-1]])/np.max(y), where='post', *args, **kwargs)[0]

def plot_spec(num, fill, *args, **kwargs): 
    y = specs.iloc[num].values
    flux = y*np.diff(gs)/np.diff(ln(gs))
    if not fill:
        return step_plot(flux, *args, **kwargs)
    else:
        line = ary([flux, flux]).T.flatten()/flux.max()
        xcoords = ary([gs[:-1], gs[1:]]).T.flatten()
        return plt.fill_between(xcoords, line, *args, **kwargs)

def plot_response(num, *args, **kwargs): 
    y = R_df.iloc[num].values
    return step_plot(y, *args, **kwargs)

if __name__=='__main__':
    if sys.argv[-1]=='0':
        plot_spec(7, fill=False, label=r'rescaled DD spectrum ($\phi$)', color='C0') # Co neutron spectra is fucking useless
        plot_spec(1, fill=False, label=r'rescaled DT spectrum ($\phi$)', color='C1')

        plt.xscale('log')
        plt.xlabel('E (eV)')
        plt.ylabel('fluence = 1/[Area]')
        plt.yticks([0,1], ['0', 'max'])
        # plt.title(r'$\int_0^{\infty} \sigma(E) \phi(E) dE$ = reaction rates')
        plt.legend()
        plt.xlim([1E2,1E8])
        # plt.yscale('log')
        plt.tight_layout(True)
        plt.savefig('temp.png', figsize=[5,3.6], dpi=120)

    elif sys.argv[-1]=='1':
        plot_spec(7, fill=True, label=r'rescaled DD spectrum ($\phi$)') # Co neutron spectra is fucking useless

        plot_response(1, label=r'rescaled cross-section $\sigma$'+':\n'+R_df.index[1].replace('_',',').replace(',g)',r',$\gamma$)'), color='C1') # Co neutron capture releasing gamma
        plot_response(2, label=r'rescaled cross-section $\sigma$'+':\n'+R_df.index[2].replace('_',',').replace(',g)',r',$\gamma$)'), color='C2') # Fe 54 n,p threshold reaction

        plt.xscale('log')
        plt.xlabel('E (eV)')
        plt.ylabel('fluence = 1/[Area]\n response = [Area]')
        plt.yticks([0,1], ['0', 'max'])
        plt.title('measured radioactivity\n'+r'$\propto \int_0^{\infty} \sigma(E) \phi(E) dE$')
        plt.legend()

        plt.tight_layout(True)
        plt.savefig('temp.png', figsize=[5,3.6], dpi=200)

    elif sys.argv[-1]=='2':
        plot_spec(7, fill=True, label=r'rescaled DD spectrum ($\phi$)') # Co neutron spectra is fucking useless

        plot_response(1, label=r'rescaled cross-section $\sigma$'+':\n'+R_df.index[1].replace('_',',').replace(',g)',r',$\gamma$)'), color='C1') # Co neutron capture releasing gamma
        plot_response(2, label=r'rescaled cross-section $\sigma$'+':\n'+R_df.index[2].replace('_',',').replace(',g)',r',$\gamma$)'), color='C2') # Fe 54 n,p threshold reaction

        plt.xscale('log')
        plt.xlabel('E (eV)')
        plt.ylabel('fluence = 1/[Area]\n response = [Area]')
        plt.yticks([0,1], ['0', 'max'])
        plt.title('measured radioactivity\n'+r'$\propto \int_0^{\infty} \sigma(E) \phi(E) dE$')
        plt.legend()

        plt.tight_layout(True)

        #label divide into coarse groups
        plt.axvspan(1E3, 1E5, facecolor='grey', zorder=0)
        plt.text(1E0, 0.5, 'thermal', ha='center')
        plt.text(1E4, 0.5, 'epithermal', ha='center')
        plt.text(1E6, 0.5, 'fast', ha='center')
        plt.savefig('temp.png', figsize=[5,3.6], dpi=200)

    elif sys.argv[-1]=='3': # plot unfolded result

        plt.plot([gs[0], 1E3,1E3, 1E5, 1E5, gs[-1]], [0.05, 0.05,0,0,0.5,0.5], label='unfolded solution')
        plt.xscale('log')
        plt.xlabel('E (eV)')
        plt.ylabel('fluence = 1/[Area]')
        plt.yticks([0,1], ['0', 'max'])
        plt.title('unfolded result')
        plt.legend()

        plt.tight_layout(True)

        #label divide into coarse groups
        plt.axvspan(1E3, 1E5, facecolor='grey', zorder=0)
        plt.text(1E0, 0.5, 'thermal', ha='center')
        plt.text(1E4, 0.5, 'epithermal', ha='center')
        plt.text(1E6, 0.5, 'fast', ha='center')
        plt.savefig('temp.png', figsize=[5,3.6], dpi=200)
