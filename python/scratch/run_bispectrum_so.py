'''

'''

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import os
import numpy as np
from scipy.special import spherical_jn
sys.path.insert(0,'./../')
import fisher


def run(prim_template):

    lmin = 2
    lmax = 5000

    opj = os.path.join
    ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/'

    camb_opts = dict(camb_out_dir = opj(ana_dir, 'camb_output/high_acy/sparse_5000'),
                     tag='r0',
                     lensed=False,
                     high_ell=True)

    F = fisher.Fisher()
    F.get_camb_output(**camb_opts)
    F.init_bins(lmin=lmin, lmax=lmax, 
                parity='odd', verbose=True)
    F.init_wig3j()
    bins = F.bins

    radii = F.get_updated_radii()

    if prim_template == 'local':
        f, _ = F.local()
    else:
        f, _ = F.equilateral()

    F.beta(f=f, radii=radii, verbose=True)
    F.init_pol_triplets()
    if F.mpi_rank  == 0:
        print 'done precalculating'

    L_tups = [(+1, +1, +1),
              (+1, -1, -1),
              (-1, +1, +1),
              (-1, -1, -1),
              (+1, -1, +1),
              (-1, +1, -1)]

    for Lidx, L_tup in enumerate(L_tups):

        if F.mpi_rank == 0:
            print 'working on DL1, DL2, DL3:', L_tup

        if Lidx == 0:
            B = F.binned_bispectrum(*L_tup, prim_template=prim_template)
        else:
            B += F.binned_bispectrum(*L_tup, prim_template=prim_template)

    if F.mpi_rank  == 0:

        np.save(opj(ana_dir, 'bispectrum/run_so', prim_template,
                    'bispectrum.npy'), B)
        np.save(opj(ana_dir, 'bispectrum/run_so', prim_template,
                    'bins.npy'), F.bins)
        np.save(opj(ana_dir, 'bispectrum/run_so', prim_template,
                    'num_pass.npy'), F.num_pass)
        np.save(opj(ana_dir, 'bispectrum/run_so', prim_template,
                    'first_pass.npy'), F.first_pass)
        np.save(opj(ana_dir, 'bispectrum/run_so', prim_template,
                    'beta_s.npy'), F.depo['scalar']['b_beta'])
        np.save(opj(ana_dir, 'bispectrum/run_so', prim_template,
                    'beta_t.npy'), F.depo['tensor']['b_beta'])
        np.save(opj(ana_dir, 'bispectrum/run_so', prim_template,
                    'pol_trpl.npy'), F.pol_trpl)

if __name__ == '__main__':

#    prim_template = 'local'
    prim_template = 'equilateral'
#    prim_template = 'orthogonal'
    run(prim_template)
