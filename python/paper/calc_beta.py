'''
Calculate and save beta.
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
import fisher

opj = os.path.join

pr = cProfile.Profile()


def run(prim_template='equilateral', out_dir=None, camb_dir=None, lmin=2, lmax=1000):
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

    F = fisher.Fisher()

    if F.mpi_rank == 0:
        if not os.path.exists(out_dir):
            raise IOError('{} does not exist'.format(out_dir))
        if not os.path.exists(opj(out_dir, prim_template)):
            os.makedirs(opj(out_dir, prim_template))
            
    F.get_camb_output(**camb_opts)
    F.init_bins(lmin=lmin, lmax=lmax, 
                parity='odd', verbose=True)

    radii = F.get_updated_radii()
    radii = radii[::5]

#    pr.enable()
    F.beta(func=prim_template, radii=radii, verbose=True, optimize=True,
           bin=False, interp_factor=40)
#    pr.disable()
#    ps = pstats.Stats(pr, stream=sys.stdout)
#    ps.sort_stats('cumulative')
#    ps.print_stats()

    if F.mpi_rank  == 0:

        np.save(opj(out_dir, prim_template, 'beta_s.npy'), 
                F.depo['scalar']['beta'])
        np.save(opj(out_dir, prim_template, 'beta_t.npy'), 
                F.depo['tensor']['beta'])

        np.save(opj(out_dir, prim_template, 'radii.npy'), 
                F.depo['radii'])

if __name__ == '__main__':

    base_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/'

    # out_dir = opj(base_dir, '20180911_sst/beta')
    out_dir = opj(base_dir, '20180911_sst/beta_sparse_ell')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/sparse_5000')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/nolens_4000')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/nolens_5200')
    camb_dir = opj(base_dir, '20180911_sst/camb_output/lensed_r0_4000')

    run(out_dir=out_dir, camb_dir=camb_dir, lmax=1000)

