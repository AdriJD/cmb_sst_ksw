'''
Calculate and plot Fisher information over subsets of r.
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
from sst import Fisher

opj = os.path.join

def run(out_dir, num_chunks, camb_opts=None,
               bin_opts=None, 
               beta_opts=None):

    F = Fisher(out_dir)

    F.get_camb_output(**camb_opts)
    F.get_bins(**bin_opts)

    radii = F.get_updated_radii()

    F.get_beta(radii=radii, **beta_opts)

    chunks = np.array_split(radii, num_chunks)
    for cidx, chunk in enumerate(chunks):
        F.get_binned_bispec('equilateral', radii_sub=chunk,
                            load=True, tag=str(cidx))

if __name__ == '__main__':

    base_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/'

    out_dir = opj(base_dir, '20181211_sst_fisher_r')
    camb_dir = opj(base_dir, '20180911_sst/camb_output/lensed_r0_4000')
                            
    num_chunks = 20
    camb_opts = dict(camb_out_dir = camb_dir,
                     tag='',
                     lensed=False,
                     high_ell=False,
                     interp_factor=None)

    bin_opts = dict(lmin=2, lmax=4000, load=True,
                    parity='odd', verbose=True)

    beta_opts = dict(func='equilateral', verbose=True, 
                     optimize=True, interp_factor=None,
                     load=True, sparse=True)

    run(out_dir, num_chunks, bin_opts=bin_opts,
        beta_opts=beta_opts, camb_opts=camb_opts)

