'''
Calculate and save bins, beta and/or bispectrum.
'''

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse

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
        Ells corresponding to cls.
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

def add_noise(cls, ells, noise_amp_temp=0, noise_amp_e=0, noise_amp_b=0,
              lmin_b=None, lmin_e=None):
    '''
    Add noise to Cls. Also used to effectively impose lmin to BB.

    Arguments
    ---------
    cls_tot : ndarray
        Cls (shape (4,lmax-1), order II, EE, BB, TE. Modified in-place.
    ells : ndarray
        Ells corresponding to cls. 

    Keyword arguments
    -----------------
    noise_amp_temp : float
        Noise ampltidue in uK arcmin for I. (default : 0)
    noise_amp_e : float
        Noise ampltidue in uK arcmin for E. (default : 0)
    noise_amp_b : float
        Noise ampltidue in uK arcmin for B. (default : 0)
    lmin_b : int
        Add 1 K arcmin noise to B below this multipole.
    lmin_e : int
        Add 1 K arcmin noise to E below this multipole.

    Notes
    -----
    Assumes TE noise is zero.
    '''

    arcmin2radians = np.pi / 180. / 60.

    if noise_amp_temp > 0:
        noise_amp_temp *= arcmin2radians
        cls[0,:] += noise_amp_temp ** 2

    if noise_amp_e > 0:
        noise_amp_e *= arcmin2radians
        cls[1,:] += noise_amp_e ** 2

    if noise_amp_b > 0:
        noise_amp_b *= arcmin2radians
        cls[2,:] += noise_amp_b ** 2
    
    if lmin_b is not None:
        idx_min = np.where(ells == lmin_b)[0][0]
        cls[2,:idx_min] += (1e6 * arcmin2radians) ** 2

    if lmin_e is not None:
        idx_min = np.where(ells == lmin_e)[0][0]
        cls[1,:idx_min] += (1e6 * arcmin2radians) ** 2

    return
        

def run(out_dir, tag, prim_template='local', add_noise_opts={}, get_cls_opts={},
        interp_fisher_opts={}, dry=False):

    '''    
    Calculate and save bins, beta and bispec. Then calculate fisher.

    Arguments
    ---------
    out_dir : str
        Output directory for Fisher.
    tag : str
        Fisher tag for output file.

    Keyword Arguments
    ---------
    prim_template : str
    dry : bool
        Do not run fisher estimate (but do everything else).
    save_cov : bool
        Save covariance and inverse convariance.

    kwargs : {add_noise_opts}, {get_cls_opts}, {interp_fisher_opts}
    '''

    F = Fisher(out_dir)

    if F.mpi_rank == 0:
        print('working on {}'.format(tag))

    if F.mpi_rank == 0:
        print(tag)

    beta_tag = 'r1_i1_l5200_16_7'
    bins_tag = '5200'
    cosmo_tag = '5200_16_7'
    bispec_tag = '5200_16_7'

    F.get_cosmo(load=True, tag=cosmo_tag, verbose=True)

    F.get_bins(load=True, parity='odd', verbose=True, tag=bins_tag, lmin=2, lmax=5000)
 
    F.get_beta(load=True, tag=beta_tag, verbose=True)

    F.get_binned_bispec(prim_template, load=True, tag=bispec_tag)
    
    cls, ells = get_cls(F.cosmo, **get_cls_opts)

    add_noise(cls, ells, **add_noise_opts)

    invcov, cov = F.get_invcov(ells, cls, return_cov=True, write=True, write_tag=tag)    

    lmax_outer = 200
    
    if not dry:
        f_i = F.interp_fisher(invcov, ells, lmin=2, lmax_outer=lmax_outer, verbose=False,
                              **interp_fisher_opts)

        if F.mpi_rank == 0:        
            print(f_i)

        r = get_cls_opts['r']
        F.save_fisher(f_i, r=r, tag=tag)
    
    return

def cv_scaling(out_dir, prim_template='local', A_lens=0.1, r=0.):
    '''
    Arguments
    ---------
    out_dir : str
        Output directory for Fisher.

    Keyword Arguments
    ---------
    prim_template : str    
    '''

    #lmax_start = 500
    #lmax_end = 4900
    lmax_start = 10
    lmax_end = 50

    #lmax_steps = 50
    #lmax_arr =  np.logspace(np.log10(lmax_start), np.log10(lmax_end), lmax_steps)
    lmax_arr = np.arange(lmax_start, lmax_end + 1)
    lmax_arr = lmax_arr.astype(int)

    #lmin_b_arr = np.asarray([2, 20, 30, 50, 80])
    lmin_b_arr = np.asarray([2])

    pol_opts_arr = [dict(no_ee=False, no_tt=False),
                    dict(no_ee=True, no_tt=False),
                    dict(no_ee=False, no_tt=True)]


    noise_amp_temp = 0
    noise_amp_e = 0
    noise_amp_b = 0
    lmin_e = 2

    for lmax in lmax_arr:
        for lmin_b in lmin_b_arr:
            for pol_opts in pol_opts_arr:

                add_noise_opts = dict(noise_amp_temp=noise_amp_temp,
                                      noise_amp_e=noise_amp_e,
                                      noise_amp_b=noise_amp_b,
                                      lmin_b=lmin_b,
                                      lmin_e=lmin_e)

                no_ee = pol_opts['no_ee']
                no_tt = pol_opts['no_tt']

                get_cls_opts = dict(A_lens=A_lens,
                                    r=r,
                                    no_ee=no_ee,
                                    no_tt=no_tt)

                interp_fisher_opts = dict(lmax=lmax)

                tag = ('{}_nt{:.4f}_ne{:.4f}_nb{:.4f}_lb{:d}_le{:d}_nee{:d}'
                       '_ntt{:d}_a{:.4f}_r{:.4f}_l{:d}'.format(prim_template,
                    noise_amp_temp, noise_amp_e, noise_amp_b, lmin_b,
                    lmin_e, int(no_ee), int(no_tt), A_lens, r, lmax))

                save_fisher_opts = dict(tag=tag)

                run(out_dir, tag, prim_template=prim_template,
                    add_noise_opts=add_noise_opts,
                    get_cls_opts=get_cls_opts,
                    interp_fisher_opts=interp_fisher_opts)

def pol(out_dir, prim_template='local', A_lens=0.5, r=0.001, plot=False, dry=False):
    '''
    Arguments
    ---------
    out_dir : str
        Output directory for Fisher.

    Keyword Arguments
    ---------
    prim_template : str    
    '''

    lmax_start = 500
    lmax_end = 4900
    lmax_steps = 10
    lmax_arr =  np.logspace(np.log10(lmax_start), np.log10(lmax_end), lmax_steps)
    lmax_arr = lmax_arr.astype(int)

    lmin_b_arr = np.asarray([2, 20, 30, 50, 80])

    pol_opts_arr = [dict(no_ee=False, no_tt=False),
                    dict(no_ee=True, no_tt=False),
                    dict(no_ee=False, no_tt=True)]


    noise_amp_temp = 4
    noise_amp_e = 4 * np.sqrt(2)
    noise_amp_b = 4 * np.sqrt(2)
    lmin_e = 2

    for lmax in lmax_arr:
        for lmin_b in lmin_b_arr:
            for pol_opts in pol_opts_arr:

                add_noise_opts = dict(noise_amp_temp=noise_amp_temp,
                                      noise_amp_e=noise_amp_e,
                                      noise_amp_b=noise_amp_b,
                                      lmin_b=lmin_b,
                                      lmin_e=lmin_e)

                no_ee = pol_opts['no_ee']
                no_tt = pol_opts['no_tt']

                get_cls_opts = dict(A_lens=A_lens,
                                    r=r,
                                    no_ee=no_ee,
                                    no_tt=no_tt)

                interp_fisher_opts = dict(lmax=lmax)

                tag = ('{}_nt{:.4f}_ne{:.4f}_nb{:.4f}_lb{:d}_le{:d}_nee{:d}'
                       '_ntt{:d}_a{:.4f}_r{:.4f}_l{:d}'.format(prim_template,
                    noise_amp_temp, noise_amp_e, noise_amp_b, lmin_b,
                    lmin_e, int(no_ee), int(no_tt), A_lens, r, lmax))

                save_fisher_opts = dict(tag=tag)

                run(out_dir, tag, prim_template=prim_template,
                    add_noise_opts=add_noise_opts,
                    get_cls_opts=get_cls_opts,
                    interp_fisher_opts=interp_fisher_opts,
                    dry=dry)

def noise(out_dir, prim_template='local', r=0.001, lmax=4900, lmin_b=50, A_lens=None):
    '''
    Arguments
    ---------
    out_dir : str
        Output directory for Fisher.

    Keyword Arguments
    ---------
    prim_template : str    
    '''

    pol_opts = dict(no_ee=False, no_tt=False)
    lmin_e = 2

    noise_i_arr = [0.3, 1, 3, 10]
    noise_b_arr = np.logspace(np.log10(0.3), np.log10(50), 10)
    if A_lens is not None:
        A_lens_arr = [A_lens]
    else:
        A_lens_arr = [0.1, 1]

    for A_lens in A_lens_arr:
        for nb_idx, n_b in enumerate(noise_b_arr):
            for ni_idx, n_i in enumerate(noise_i_arr):

                noise_amp_temp = n_i
                noise_amp_e = n_i * np.sqrt(2)
                noise_amp_b = n_b

                add_noise_opts = dict(noise_amp_temp=noise_amp_temp,
                                      noise_amp_e=noise_amp_e,
                                      noise_amp_b=noise_amp_b,
                                      lmin_b=lmin_b,
                                      lmin_e=lmin_e)

                no_ee = pol_opts['no_ee']
                no_tt = pol_opts['no_tt']

                get_cls_opts = dict(A_lens=A_lens,
                                    r=r,
                                    no_ee=no_ee,
                                    no_tt=no_tt)

                interp_fisher_opts = dict(lmax=lmax)

                tag = ('{}_nt{:.4f}_ne{:.4f}_nb{:.4f}_lb{:d}_le{:d}_nee{:d}'
                       '_ntt{:d}_a{:.4f}_r{:.4f}_l{:d}'.format(prim_template,
                    noise_amp_temp, noise_amp_e, noise_amp_b, lmin_b,
                    lmin_e, int(no_ee), int(no_tt), A_lens, r, lmax))

                save_fisher_opts = dict(tag=tag)

                run(out_dir, tag, prim_template=prim_template,
                    add_noise_opts=add_noise_opts,
                    get_cls_opts=get_cls_opts,
                    interp_fisher_opts=interp_fisher_opts)

if __name__ == '__main__':

    base_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/'

    parser = argparse.ArgumentParser()
    parser.add_argument("odir")
    args = parser.parse_args()

    out_dir = args.odir    

    #run(out_dir)
    cv_scaling(out_dir, A_lens=0.1, r=0.1, prim_template='local')
    #pol(out_dir, A_lens=0.5, r=0.001, dry=True) 
    #noise(out_dir, A_lens=1)

