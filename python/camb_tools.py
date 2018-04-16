'''
Some functions used to read ouput from CAMB. For now, always run
it with do_lensing = T, get_tensor_cls = T, and l_sample_boost = 50
'''

import numpy as np
from scipy.io import FortranFile
import os
import sys

opj = os.path.join

def read_camb_output(source_dir, ttype='scalar'):
    '''
    Read in the transfer functions and aux files
    outputted by camb using scipy's FortranFile.

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
    '''
    
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
    ell = f.read_reals(np.int32)
    f.close()

    # trim trailing zeros that camb puts there
    ell = np.trim_zeros(ell, trim='b')
    lmax = ell[-1]

    # read ks
    f = FortranFile(opj(source_dir, k_name), 'r')
    k = f.read_reals(float)
    f.close()

    # read num sources
    f = FortranFile(opj(source_dir, numsources_name), 'r')
    num_sources = f.read_reals(np.int32)
    f.close()
    num_sources = num_sources[0]
    print 'num_sources', num_sources
    # reshape and turn to c-contiguous 
    transfer = delta_p_l_k.reshape((num_sources, ell.size, k.size), order='F')
    transfer = np.ascontiguousarray(transfer)

    # add monopole and dipole
#    transfer_full = np.zeros((num_sources, lmax+1, k.size), dtype=transfer.dtype)
#    transfer_full[:,2:,:] = transfer

    return transfer, lmax, k

def get_spectra(source_dir, tag='', lensed=True):
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

    Returns
    -------
    cls : array-like
        TT, EE, BB, TE spectra, shape: (4, lmax-1). Note, start from ell=2
    lmax : int
    '''

    if lensed:
        camb_name = 'lensedtotCls.dat'
        # note that lensedtotCls also includes tensor contribution, lensedCl is just scalar
    else:
        camb_name = 'totCls.dat'

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

def get_so_noise(tt_file, pol_file):

    nl_tt = np.loadtxt(tt_file)
    nl_tt = nl_tt.transpose()
    nl_tt = np.ascontiguousarray(nl_tt)

    ell_tt = nl_tt[0]
    nl_tt = nl_tt[1]

    nl_pol = np.loadtxt(pol_file)
    nl_pol = nl_pol.transpose()
    nl_pol = np.ascontiguousarray(nl_pol)

    ell_pol = nl_pol[0]
    nl_pol = nl_pol[1]

    return ell_tt, nl_tt, ell_pol, nl_pol
    
    
#cls = get_spectra('/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output', tag='test', lensed=False)  
#print cls[2,:]

#tr, lmax, k = read_camb_output('/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output', ttype='tensor')
#print tr.shape
#print k.shape
#print k
#print lmax
#print tr.flags
#print tr[0], tr[1], tr[2]
