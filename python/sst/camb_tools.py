'''
Some functions used to read ouput from CAMB. For now, always run
it with do_lensing = T, get_tensor_cls = T, and l_sample_boost = 50
'''

import numpy as np
from scipy.io import FortranFile
import os
import sys
import camb
import tools

opj = os.path.join

def run_camb(lmax, k_eta_fac=5, AccuracyBoost=3, lSampleBoost=2, 
             lAccuracyBoost=2, verbose=True):
    '''
    Run camb to get transfer functions and power spectra.

    Arguments
    ---------
    lmax : int
        Max multipole used in calculation transfer functions.

    Keyword arguments
    -----------------
    k_eta_fac : scalar
        Determines k_eta_max for transfer functions through
        k_eta_max = k_eta_fac * lmax. (default : 5)
    AccuracyBoost : int
        Overall accuracy CAMB, pick 1 to 3. (default : 3)
    lSampleBoost : int
        Factor determining multipole sampling of transfer 
        functions, pick 1 to 50. (default : 2)
    lAccuracyBoost : int
        Factor determining truncation of Boltzmann hierarchy,
        pick 1 to 3 (default : 2)
    verbose : bool

    Returns
    -------
    transfer : dict
        Contains transfer functions, wavenumbers, multipoles.
        Transfer function have shape (numsources, lmax-1, k_size)
        numsources is 3 for lensed scalar (T, E, P) and 3 for
        tensor (T, E, B).
    cls : dict
        Contains power spectra and corresponding multipoles.
            cls : dict
                TT, EE, BB, TE spectra, shape: (4, lmax-1)
            ells : ndarray
    opts : dict
        Contains accuracy, cosmology and primordial
        options.
    '''

    transfer = {}
    cls = {}
    opts = {}

    # Accuracy options.
    acc_opts = dict(AccuracyBoost=AccuracyBoost,
                    lSampleBoost=lSampleBoost, 
                    lAccuracyBoost=lAccuracyBoost, 
                    DoLateRadTruncation=False)



    # Hardcoded LCDM+ parameters.
    cosmo_opts = dict(H0=67.66, 
                       TCMB=2.7255,
                       YHe=0.24,
                       standard_neutrino_neff=True,
                       ombh2=0.02242, 
                       omch2=0.11933, 
                       tau=0.0561,
                       mnu=0.06, 
                       omk=0)

    prim_opts = dict(ns=0.9665,
                     r=1.,
                     pivot_scalar=0.05,
                     As=2.1056e-9,
                     nt=0, 
                     parameterization=2)

    pars = camb.CAMBparams()
    pars.set_cosmology(**cosmo_opts)
    pars.InitPower.set_params(**prim_opts)
    pars.set_accuracy(**acc_opts)

    pars.WantScalars = True
    pars.WantTensors = True

    # CAMB becomes unstable for too low ell and k.
    lmax = max(300, lmax)
    max_eta_k = k_eta_fac * lmax
    max_eta_k = max(max_eta_k, 1000)
    
    pars.max_l = lmax
    pars.max_l_tensor = lmax
    pars.max_eta_k = max_eta_k
    pars.max_eta_k_tensor = max_eta_k

    pars.AccurateBB = True
    pars.AccurateReionization = True
    pars.AccuratePolarization = True


    acc_opts['k_eta_fac'] = k_eta_fac
    opts['acc'] = acc_opts
    opts['cosmo'] = cosmo_opts
    opts['prim'] = prim_opts
    
    if pars.validate() is False:
        raise ValueError('Invalid CAMB input')

    if verbose:
        print('Calculating transfer functions\n')
        print('lmax: {} \nmax_eta_k : {} \nAccuracyBoost : {} \n'
              'lSampleBoost : {} \nlAccuracyBoost : {}\n'.format(
                  lmax, max_eta_k, AccuracyBoost, lSampleBoost, 
                  lAccuracyBoost))

    data = camb.get_transfer_functions(pars)
    transfer_s = data.get_cmb_transfer_data('scalar')
    transfer_t = data.get_cmb_transfer_data('tensor')

    if verbose:
        print('Calculating power spectra')

    # NOTE that l=0, l=1 are also included here.
    data.calc_power_spectra()
    cls_camb = data.get_cmb_power_spectra(lmax=None, CMB_unit='muK', 
                                          raw_cl=True)

    # CAMB cls are column-major, so convert.
    # NOTE this is wrong. You need to do transpose + ascontingousarray
    for key in cls_camb:
        cls_cm = cls_camb[key]
        n_ell, n_pol = cls_cm.shape # 2d tuple.
        cls_camb[key] = cls_cm.reshape(n_pol, n_ell)

    ells_cls = np.arange(2, n_ell + 2)
    cls['ells'] = ells_cls
    cls['cls'] = cls_camb

    # We need to modify scalar E-mode and tensor I transfer functions, 
    # see Zaldarriaga 1997 eq. 18 and 39. (CAMB applies these factors
    # at a later stage).
    ells = transfer_s.l
    # CAMB ells are in int32, gives nan in sqrt, so convert first.
    ells = ells.astype(int) 
    prefactor = np.sqrt((ells + 2) * (ells + 1) * ells * (ells - 1))
    
    transfer_s.delta_p_l_k[1,...] *= prefactor[:,np.newaxis]
    transfer_t.delta_p_l_k[0,...] *= prefactor[:,np.newaxis]

    # both scalar and tensor have to be scaled with monopole in uK
    transfer_s.delta_p_l_k *= (pars.TCMB * 1e6)
    transfer_t.delta_p_l_k *= (pars.TCMB * 1e6)

    # Scale tensor transfer by 1/4 to correct for CAMB output.
    transfer_t.delta_p_l_k /= 4.

    # Check for nans.
    if tools.has_nan(transfer_s.delta_p_l_k):
        raise ValueError('nan in scalar transfer')
    if tools.has_nan(transfer_t.delta_p_l_k):
        raise ValueError('nan in tensor transfer')
    if tools.has_nan(transfer_s.q):
        raise ValueError('nan in k array')

    transfer['scalar'] = transfer_s.delta_p_l_k
    transfer['tensor'] = transfer_t.delta_p_l_k
    transfer['k'] = transfer_s.q
    transfer['ells'] = ells # sparse and might differ from cls['ells']

    print(transfer['k'].shape)
    print(transfer['tensor'].shape)

    return transfer, cls, opts

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
    ells : array-like
        Multipoles used by CAMB (sparse when high_ell is set)
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
    # Convert to 64 bit to avoid overflow later.
    ells = ells.astype(np.int64)

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
    num_sources = num_sources.astype(np.int64)
    num_sources = num_sources[0]

    # reshape and turn to c-contiguous 
    transfer = delta_p_l_k.reshape((num_sources, ells.size, k.size), order='F')
    transfer = np.ascontiguousarray(transfer)

    # Correct and scale transfer here.
    prefactor = np.sqrt((ells + 2) * (ells + 1) * ells * (ells - 1))
    # We need to multiply scalar e-mode and tensor I transfer, 
    # see Zaldarriaga 1997 eq. 18 and 39. (CAMB applies these factors
    # at a later stage).
    if ttype == 'scalar':
        transfer[1,...] *= prefactor[:,np.newaxis]
    elif ttype == 'tensor':
        transfer[0,...] *= prefactor[:,np.newaxis]

    # both scalar and tensor have to be scaled with monopole in uK
    transfer *= 2.7255e6

    # Scale tensor transfer by 1/4 to correct for CAMB output.
    if ttype == 'tensor':
#        transfer /= (4. * np.pi)
        transfer /= 4.

    # dont interpolate transfer, it fluctuates too fast
    # instead, use sparsely sampled high ell part for beta
    # let beta bin using sparse ells
    # but ideally, use full transfer for 2 <= ell <= 4000 part

    return transfer, lmax, k, ells
    
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

