'''
Calculate and plot bispectrum slice.
'''

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import cProfile, pstats

import sys
import os
import numpy as np
from scipy.special import spherical_jn
sys.path.insert(0,'./../')
from sst import Fisher, camb_tools, plot_tools, tools

opj = os.path.join

def run(ell, prim_template='local', out_dir=None,
        lmin=2, lmax=120):
    '''    
    Calculate and save bins, beta and bispec. Then return bispec slice.

    Keyword Arguments
    ---------
    ell : int
        Multipole for bispec slice.
    prim_template : str
    out_dir : str
    lmin : int
    lmax : int
    '''

    interp_factor = 10
    ac = 16
    ke = 10

#    tag = 'l{}'.format(lmax)
#    beta_tag = 'l{}_i{}'.format(lmax, interp_factor)
#    cosmo_tag = 'l{}_ac{}_ke{}'.format(lmax, ac, ke)

    bins_tag = '5200'
    beta_tag = 'r1_i1_l{}_16_7'.format(lmax, interp_factor)
    cosmo_tag = '5200_16_7'.format(lmax, ac, ke)
    bispec_tag = '5200_16_7'

    F = Fisher(out_dir)

    F.get_cosmo(load=True, tag=cosmo_tag, verbose=True, lmax=lmax * 2, 
                AccuracyBoost=ac, k_eta_fac=ke)
    F.get_bins(load=True, parity='odd', verbose=True, tag=bins_tag, lmin=lmin, lmax=lmax)

    F.get_beta(load=True, tag=beta_tag, verbose=True, interp_factor=interp_factor)

    F.get_binned_bispec(prim_template, load=True, tag=bispec_tag)
    
    b_slice, doublets = F.get_bispec_slice(ell, verbose=2)
    
    return F, b_slice, doublets

def plot_slice(F, filename, b_slice, doublets, plot_lmin=None, plot_lmax=None):
    '''

    '''
    if F.mpi_rank == 0:
        pol_trpl = F.bispec['pol_trpl']
        if np.sum(np.isnan(b_slice)):
            # Panic.
            raise ValueError('nan in b_slice')
        plot_tools.plot_bispec_slice(filename, doublets, b_slice, pol_trpl,
                                     plot_lmin=plot_lmin, plot_lmax=plot_lmax)
        
    else:
        pass
    F.barrier()
    return


def plot_binned_slice(F, filename, plot_lmin=None, plot_lmax=None):
    '''

    '''
    if F.mpi_rank == 0:
        pol_trpl = F.bispec['pol_trpl']
        bins = F.bins['bins']
        bidx = tools.ell2bidx(ell, bins)
        
        bispec = F.bispec['bispec']    
        bispec = bispec[bidx,:,:,:]
        bispec = bispec * F.common_amp
        # No factor of num pass in bispec anymore.

        # Create doublets.
        a, b = np.meshgrid(bins, bins)
        doublets = np.empty((a.size + b.size), dtype=int)
        doublets[0::2] = b.ravel()
        doublets[1::2] = a.ravel()
        doublets = doublets.reshape(a.size, 2)
        
        s = bispec.shape
        b_slice = bispec.reshape(s[0] * s[1], s[2])
                
        plot_tools.plot_bispec_slice(filename, doublets, b_slice, pol_trpl,
                                     plot_lmin=plot_lmin, plot_lmax=plot_lmax)
        
    else:
        pass
        
    F.barrier()
    return

if __name__ == '__main__':

    base_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/'
#    out_dir = opj(base_dir, '20181219_sst_interp')
    out_dir = opj(base_dir, '20190426_sst_plot')

    ell = 40
    lmax = 5200

    plot_lmin = 40
    plot_lmax = 100

    fformat = 'png'

    prim_template = 'local'
    F, b_slice, doublets = run(ell, out_dir=out_dir, lmax=lmax,
                               prim_template=prim_template)

    filename = opj(out_dir, 'img', 'bispec_{}_ell{}_lmax{}.{}'.format(
                                     prim_template, ell, lmax, fformat))
    plot_slice(F, filename, b_slice, doublets, plot_lmin=plot_lmin, plot_lmax=plot_lmax)

    filename_b = opj(out_dir, 'img', 'bispec_{}_b_ell{}_lmax{}.{}'.format(
                                         prim_template, ell, lmax, fformat))
    plot_binned_slice(F, filename_b, plot_lmin=plot_lmin, plot_lmax=plot_lmax)
