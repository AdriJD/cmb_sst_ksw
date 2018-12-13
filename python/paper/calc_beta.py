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

def run(prim_template='equilateral', out_dir=None, camb_dir=None, lmin=2, lmax=5000):
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

    camb_opts = dict(camb_out_dir = camb_dir,
                     tag='',
                     lensed=False,
                     high_ell=False,
                     interp_factor=None)

    F = Fisher(out_dir)

    F.get_camb_output(**camb_opts)
    F.get_bins(lmin=lmin, lmax=lmax, load=True,
                parity='odd', verbose=True)

    interp_factor = 1
    radii_factor = 40
    beta_tag = 'r{}_i{}_l{}'.format(radii_factor, interp_factor, lmax)

    radii = F.get_updated_radii()
    radii = radii[::radii_factor]


    F.get_beta(func=prim_template, radii=radii, verbose=True, optimize=True,
               interp_factor=interp_factor, load=True, sparse=True, tag=beta_tag)

    F.get_binned_bispec('equilateral', load=True)
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
    out_dir = opj(base_dir, '20181123_sst')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/sparse_5000')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/nolens_4000')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/nolens_5200')
    camb_dir = opj(base_dir, '20180911_sst/camb_output/lensed_r0_4000')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/sparse_5000')

    run(out_dir=out_dir, camb_dir=camb_dir, lmax=400)

