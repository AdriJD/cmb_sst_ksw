'''
Some functions used to read ouput from CAMB. For now, always run
it with do_lensing = T, get_tensor_cls = T, and l_sample_boost = 50
'''

import numpy as np
from scipy.io import FortranFile
import os
import sys

opj = os.path.join

def read_camb_output(source_dir, ttype='scalar', high_ell=False):
    '''
    Read in the transfer functions and aux files
    outputted by camb using scipy's FortranFile.

    Interpolates if CAMB sampled ells sparsely.

    Arguments
    ---------
    source_dir : str
        Path to files

    Keyword arguments
    -----------------
    ttype : str
        Transfer type, either "scalar" or "tensor"
        (default : "scalar")

    Returns
    -------
    transfer : array-like
        Transfer function with shape (numsources, lmax+1, k_size)
        numsources is 2 voor unlensed scalar (T, E), 3 for lensed
        scalar (T, E, P) and 3 for tensor (T, E, B)
    lmax : int
        Maximum multipole
    k : array-like
        array with wavenumbers in Mpc^-1
    ell : array-like
        Sparsely sampled ells array (only when high_ell is set)
    '''
    
    if high_ell:
        transfer_name = 'Delta_p_l_k_{}_hl.dat'.format(ttype)
        ell_name = 'l_{}_hl.dat'.format(ttype)
        k_name = 'points_{}_hl.dat'.format(ttype)
        numsources_name = 'NumSources_{}_hl.dat'.format(ttype)

    else:
        transfer_name = 'Delta_p_l_k_{}.dat'.format(ttype)
        ell_name = 'l_{}.dat'.format(ttype)
        k_name = 'points_{}.dat'.format(ttype)
        numsources_name = 'NumSources_{}.dat'.format(ttype)

    # read transfer
    f = FortranFile(opj(source_dir, transfer_name), 'r')
    delta_p_l_k = f.read_reals(float)
    f.close()
    
    # read ells
    f = FortranFile(opj(source_dir, ell_name), 'r')
    ells = f.read_reals(np.int32)
    f.close()

    # trim trailing zeros that camb puts there
    ells = np.trim_zeros(ells, trim='b')
    lmax = ells[-1]

    # read ks
    f = FortranFile(opj(source_dir, k_name), 'r')
    k = f.read_reals(float)
    f.close()

    # read num sources
    f = FortranFile(opj(source_dir, numsources_name), 'r')
    num_sources = f.read_reals(np.int32)
    f.close()
    num_sources = num_sources[0]

    # reshape and turn to c-contiguous 
    transfer = delta_p_l_k.reshape((num_sources, ells.size, k.size), order='F')
    transfer = np.ascontiguousarray(transfer)

    # correct and scale transfer here
    dells = ells * (ells + 1)
    # we need to multiply scalar e-mode and tensor I transfer
    if ttype == 'scalar':
        transfer[1,...] *= dells[:,np.newaxis]
    elif ttype == 'tensor':
        transfer[0,...] *= dells[:,np.newaxis]

    # both scalar and tensor have to be scaled with monopole in uK
    transfer *= 2.7255e6

    # we scale tensor transfer by 1/(4pi) to match shiraishi's definition
    if ttype == 'tensor':
        transfer /= (4. * np.pi)

    # dont interpolate transfer, it fluctuates too fast
    # instead, use sparsely sampled high ell part for beta
    # let beta bin using sparse ells
    # but ideally, use full transfer for 2 <= ell <= 4000 part

    if high_ell:
        # also return ell, because it's non-trivial in this case
        return transfer, lmax, k, ells

    else:
        return transfer, lmax, k

def get_spectra(source_dir, tag='', lensed=True, prim_type='tot'):
    '''
    Read camb output and return TT, EE, BB, TE spectra.
    Units are uK^2
    
    Arguments
    ---------
    source_dir : str
        Look for camb output in <source_dir>/
        
    Keyword arguments
    -----------------
    tag : str
        Use files that match: <tag>_
        (default : '')
    lensed : bool
        Whether spectra contain lensed contributions
        (default : True)
    prim_type : str
        Either 'scalar', 'tensor' or 'tot'

    Returns
    -------
    cls : array-like
        TT, EE, BB, TE spectra, shape: (4, lmax-1). Note, start from ell=2
    lmax : int
    '''

    if lensed and prim_type != 'tot':
        # I need to think if this is fine, for now: raise error
        raise ValueError('wrong comb')

    if lensed:
        camb_name = 'lensedtotCls.dat'
        # note that lensedtotCls also includes tensor contribution, 
        # lensedCl is just scalar
    else:
        if prim_type == 'tot':
            camb_name = 'totCls.dat'

        elif prim_type == 'scalar':
            camb_name = 'scalCls.dat'

        elif prim_type == 'tensor':
            camb_name = 'tensCls.dat'

    cls_in = np.loadtxt(opj(source_dir, '{}_{}'.format(tag, camb_name)))

    # convert to c-contiguous
    cls_in = cls_in.transpose()
    cls_in = np.ascontiguousarray(cls_in)

    lmax = cls_in.shape[1] + 1
    
    # discard ell column
    ell = cls_in[0,:]
    cls = cls_in[1:,:]
    
    # turn Dls into Cls
    cls /= (ell * (ell + 1) / 2. / np.pi)
    
    return cls, lmax

def get_so_noise(tt_file=None, pol_file=None, sat_file=None):
    '''
    Read in the SO noise curves

    Arguments
    ---------
    tt_file : str
        Path to txt file
    pol_file : str
        Path to txt file

    Returns
    -------
    ell_tt : array-like
        ell array for TT 
    nl_tt : array-like
        noise array for TT, same shape as ell_tt
    ell_pol : array-like
        ell array for EE and BB
    nl_EE : array-like
        noise array for EE, same shape as ell_pol
    nl_BB : array-like
        noise array for BB, same shape as ell_pol
        
    optional:
    ell_sat : array_like
        ell array for sat
    nl_sat : array-like
        Noise array for small-aperature telescope
        same shape as ell_sat


    Notes
    -----
    Assumes that TT textfile has colums as: ell, TT, yy
    and pol txt file has ell, EE, BB. Ell is in steps of 1

    columns colin: [ell] [N_ell^TT in uK^2] [N_ell^yy (dimensionless)]
    '''
    
    nl_tt = np.loadtxt(tt_file)
    nl_tt = nl_tt.transpose()
    nl_tt = np.ascontiguousarray(nl_tt)

    ell_tt = nl_tt[0]
    nl_tt = nl_tt[1]

    nl_pol = np.loadtxt(pol_file)
    nl_pol = nl_pol.transpose()
    nl_pol = np.ascontiguousarray(nl_pol)

    ell_pol = nl_pol[0]
    nl_ee = nl_pol[1]
    nl_bb = nl_pol[1]

    if sat_file:

        nl_sat = np.loadtxt(sat_file)
        ell_sat = np.arange(2, nl_sat.size + 2)
        
        return ell_tt, nl_tt, ell_pol, nl_ee, nl_bb, nl_sat, ell_sat

    else:
        return ell_tt, nl_tt, ell_pol, nl_ee, nl_bb

