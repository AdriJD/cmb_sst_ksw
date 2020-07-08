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
from sst import Fisher

opj = os.path.join

#def run(prim_template='equilateral', out_dir=None, camb_dir=None,
#        lmin=2, lmax=5000):
def run(prim_template='equilateral', out_dir=None,
        lmin=2, lmax=5000):

    '''    
    Calculate and save beta.

    Keyword Arguments
    ---------
    prim_template : str
    out_dir : str
    camb_dir : str
    lmin : int
    lmax : int
    '''

#    camb_opts = dict(camb_out_dir = camb_dir,
#                     tag='',
#                     lensed=False,
#                     high_ell=False,
#                     interp_factor=None)

    F = Fisher(out_dir)

    ac = 16
    ke = 10
    interp_factor = 1
    radii_factor = 1


#    F.get_camb_output(**camb_opts)
    F.get_cosmo(lmax=lmax, load=True, tag='{}_{}_{}'.format(lmax, ac, ke),
                AccuracyBoost=ac, k_eta_fac=ke)
    F.get_bins(lmin=lmin, lmax=lmax, load=True,
                parity='odd', verbose=True, tag=str(lmax))
 
    beta_tag = 'r{}_i{}_l{}_{}_{}'.format(radii_factor, interp_factor, lmax, ac, ke)

    radii = F.get_updated_radii()
    radii = radii[::radii_factor]


    F.get_beta(func=prim_template, radii=radii, verbose=True, optimize=False,
               interp_factor=interp_factor, load=False, sparse=False, tag=beta_tag)
    exit()
    F.get_binned_bispec('equilateral', load=False)
    if F.mpi_rank == 0:
        print(F.bispec['bispec'].shape)
        print(F.bispec['bispec'][F.bispec['bispec'] != 0])
        print(F.bispec['bispec'][F.bispec['bispec'] != 0].size)
        print(np.sum(F.bins['num_pass_full'].astype(bool)))
        print(F.bispec['bispec'][F.bins['num_pass_full'].astype(bool)])
        print(np.sum(F.bispec['bispec'][~F.bins['num_pass_full'].astype(bool)]))
        print(F.bispec['bispec'][np.isnan(F.bispec['bispec'])])
        print(F.bispec['bispec'][np.logical_and(F.bispec['bispec'] == 0,F.bins['num_pass_full'][:,:,:,np.newaxis].astype(bool))])
        print(F.bispec['bispec'][0,:4,:4,:])
        print(F.bins['num_pass_full'][0,:4,:4])

if __name__ == '__main__':

    base_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/'

    # out_dir = opj(base_dir, '20180911_sst/beta')
    # out_dir = opj(base_dir, '20180911_sst/beta_sparse_ell')
#    out_dir = opj(base_dir, '20181123_sst')
#    out_dir = opj(base_dir, '20181214_sst_debug')
    out_dir = opj(base_dir, '20190411_beta')

    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/sparse_5000')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/nolens_4000')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/nolens_5200')
    # camb_dir = opj(base_dir, '20180911_sst/camb_output/lensed_r0_4000')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/sparse_5000')

    run(out_dir=out_dir, lmax=2000) 

