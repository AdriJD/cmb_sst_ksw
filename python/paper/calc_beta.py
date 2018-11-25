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
from sst import Fisher

opj = os.path.join

pr = cProfile.Profile()


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
                     tag='r0',
                     lensed=False,
                     high_ell=True,
                     interp_factor=None)

    F = Fisher(out_dir)

    F.get_camb_output(**camb_opts)
    F.get_bins(lmin=lmin, lmax=lmax, load=False,
                parity='odd', verbose=True)
    exit()
    radii = F.get_updated_radii()
    radii = radii[::30]

#    pr.enable()
    F.get_beta(func=prim_template, radii=radii, verbose=True, optimize=True,
               interp_factor=None, load=False, sparse=True)
    print(F.depo['scalar']['ells_camb'])
#    pr.disable()
#    ps = pstats.Stats(pr, stream=sys.stdout)
#    ps.sort_stats('cumulative')
#    ps.print_stats()

if __name__ == '__main__':

    base_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/'

    # out_dir = opj(base_dir, '20180911_sst/beta')
    # out_dir = opj(base_dir, '20180911_sst/beta_sparse_ell')
    out_dir = opj(base_dir, '20181123_sst')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/sparse_5000')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/nolens_4000')
    # camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/nolens_5200')
    # camb_dir = opj(base_dir, '20180911_sst/camb_output/lensed_r0_4000')
    camb_dir = opj(base_dir, '20171217_sst/camb_output/high_acy/sparse_5000')

    run(out_dir=out_dir, camb_dir=camb_dir, lmax=5000)

