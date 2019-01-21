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
from sst import Fisher, camb_tools

opj = os.path.join

def get_cls(cls_path, lmax, A_lens=1, no_ee=False, no_tt=False):
    '''
    returns
    -------
    cls : array-like
        Lensed Cls (shape (4,lmax-1) with BB lensing power 
        reduced depending on A_lens. 
        order: TT, EE, BB, TE
    ells : ndarray
    '''
    
    cls_nolens, _ = camb_tools.get_spectra(cls_path, tag='',
                             lensed=False, prim_type='tot')
    cls_lensed, _ = camb_tools.get_spectra(cls_path, tag='',
                             lensed=True, prim_type='tot')

    # truncate to lmax
    cls_nolens = cls_nolens[:,:lmax-1]
    cls_lensed = cls_lensed[:,:lmax-1]

    BB_nolens = cls_nolens[2]
    BB_lensed = cls_lensed[2]
    
    # difference BB (lensed - unlensed = lens_contribution)
    BB_lens_contr = BB_lensed - BB_nolens

    # depending on A_lens, remove lensing contribution
    cls_lensed[2] -= (1. - A_lens) * BB_lens_contr
    ells = np.arange(2, lmax + 1)

    if no_ee:
        cls_lensed[1,:] = 1e48
    if no_tt:
        cls_lensed[0,:] = 1e48

    return cls_lensed, ells

def run(prim_template='equilateral', out_dir=None, camb_dir=None,
        lmin=2, lmax=5000):
    '''    
    Calculate and save bins, beta and bispec. Then calculate fisher.

    Keyword Arguments
    ---------
    prim_template : str
    out_dir : str
    camb_dir : str
    lmin : int
    lmax : int
    '''

    tag = 'l{}'.format(lmax)
    
    camb_opts = dict(camb_out_dir = camb_dir,
                     tag='',
                     lensed=False,
                     high_ell=False,
                     interp_factor=None)

    F = Fisher(out_dir)

    F.get_camb_output(**camb_opts)
    F.get_bins(lmin=lmin, lmax=lmax, load=True,
                parity='odd', verbose=True, tag=tag)

    interp_factor = 1
    radii_factor = 10
    beta_tag = 'r{}_i{}_l{}'.format(radii_factor, interp_factor, lmax)

    radii = F.get_updated_radii()
    radii = radii[::radii_factor]


    F.get_beta(func=prim_template, radii=radii, verbose=True, optimize=True,
               interp_factor=interp_factor, load=True, sparse=True, tag=beta_tag)

    F.get_binned_bispec('equilateral', load=True, tag=tag)

    cls, ells = get_cls(camb_dir, lmax, A_lens=1, no_ee=False, no_tt=False)

    invcov, cov = F.get_invcov(ells, cls, return_cov=True)    
    f_i = F.interp_fisher(invcov, ells, lmin=2, lmax=lmax, verbose=1)
    if F.mpi_rank == 0:        
        print(f_i)

    F.barrier()

    b_invcov, b_cov = F.get_binned_invcov(ells, cls, return_bin_cov=True)

    if F.mpi_rank == 0:
        f_n = F.naive_fisher(b_invcov, lmin=2, lmax=lmax)
        print(f_n)

if __name__ == '__main__':

    base_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/'

    # out_dir = opj(base_dir, '20180911_sst/beta')
    # out_dir = opj(base_dir, '20180911_sst/beta_sparse_ell')
    #out_dir = opj(base_dir, '20181123_sst')
    #out_dir = opj(base_dir, '20181214_sst_debug')
    out_dir = opj(base_dir, '20181219_sst_interp')

    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/sparse_5000')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/nolens_4000')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/nolens_5200')
    camb_dir = opj(base_dir, '20180911_sst/camb_output/lensed_r0_4000')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/sparse_5000')

    run(out_dir=out_dir, camb_dir=camb_dir, lmax=50)

