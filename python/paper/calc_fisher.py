'''
Calculate and save bins, beta and/or bispectrum.
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
from sst import Fisher, camb_tools, tools

opj = os.path.join

def get_cls(cosmo, A_lens=1, r=1, no_ee=False, no_tt=False):
    '''
    cosmo : dict
    
    Keyword arguments
    -----------------

    returns
    -------
    cls : array-like
        Lensed Cls (shape (4,lmax-1) with BB lensing power 
        reduced depending on A_lens and primordial BB scaled
        by r. Order: TT, EE, BB, TE
    ells : ndarray
    '''
    
    cls_s_nolens = cosmo['cls']['cls']['unlensed_scalar']
    cls_s_lensed = cosmo['cls']['cls']['lensed_scalar']
    cls_t = cosmo['cls']['cls']['unlensed_total']

    ells = cosmo['cls']['ells'].copy()

    # Correct for the shape of the cls.
    def correct_cls(cls):
        
        s0, s1 = cls.shape
        cls = cls.reshape(s1, s0)
        cls = np.ascontiguousarray(cls.transpose())

        return cls

    cls_s_nolens = correct_cls(cls_s_nolens)
    cls_s_lensed = correct_cls(cls_s_lensed)
    cls_t = correct_cls(cls_t)
            
    # Trim monopole, dipole.
    cls_s_nolens = cls_s_nolens[:,2:]
    cls_s_lensed = cls_s_lensed[:,2:]
    cls_t = cls_t[:,2:]

    # Truncate ells if needed.
    n_ell = cls_t.shape[1]
    ells = ells[:n_ell]

    # Start with unlensed.
    cls_tot = cls_s_nolens.copy()

    # Replace TT, EE, TE with lensed.
    cls_tot[0,:] = cls_s_lensed[0]
    cls_tot[1,:] = cls_s_lensed[1]
    cls_tot[3,:] = cls_s_lensed[3]

    # Add lensed BB scaled by A_lens
    cls_tot[2,:] += (A_lens * cls_s_lensed[2])

    # Add tensor scaled by r.
    cls_tot += (r * cls_t)

    if no_ee:
        cls_tot[1,:] = 1e48
    if no_tt:
        cls_tot[0,:] = 1e48

    return cls_tot, ells

def run(prim_template='local', out_dir=None):

    '''    
    Calculate and save bins, beta and bispec. Then calculate fisher.

    Keyword Arguments
    ---------
    prim_template : str
    out_dir : str
    lmin : int
    lmax : int
    '''

    F = Fisher(out_dir)

    beta_tag = 'r1_i1_l5200_16_7'
    bins_tag = '5200'
    cosmo_tag = '5200_16_7'
    bispec_tag = '5200_16_7'

#    beta_tag = 'r1_i1_l500_16_16'
#    bins_tag = '500'
#    cosmo_tag = '500_16_16'
#    bispec_tag = '500_16_16'

    F.get_cosmo(load=True, tag=cosmo_tag, verbose=True)

    F.get_bins(load=True, parity='odd', verbose=True, tag=bins_tag)
 
    F.get_beta(load=True, tag=beta_tag, verbose=True)

    F.get_binned_bispec(prim_template, load=True, tag=bispec_tag)
    
    r = 0.01
    A_lens = 0.1
    cls, ells = get_cls(F.cosmo, r=r, A_lens=A_lens)

    invcov, cov = F.get_invcov(ells, cls, return_cov=True)    

    lmax = 4900
    lmax_outer = 200
    f_i = F.interp_fisher(invcov, ells, lmin=2, lmax=lmax, lmax_outer=lmax_outer, 
                          verbose=2)

    if F.mpi_rank == 0:        
        print(f_i)

    F.save_fisher(f_i, r=r, tag=None)
    
    return

if __name__ == '__main__':

    base_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/'

    # out_dir = opj(base_dir, '20180911_sst/beta')
    # out_dir = opj(base_dir, '20180911_sst/beta_sparse_ell')
    #out_dir = opj(base_dir, '20181123_sst')
    #out_dir = opj(base_dir, '20181214_sst_debug')
    #out_dir = opj(base_dir, '20181219_sst_interp')
    out_dir = opj(base_dir, '20190411_beta')

    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/sparse_5000')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/nolens_4000')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/nolens_5200')
    # camb_dir = opj(base_dir, '20180911_sst/camb_output/lensed_r0_4000')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/sparse_5000')

    run(out_dir=out_dir)

