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

lmin = 2
lmax = 250

opj = os.path.join
ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/'
out_dir = opj(ana_dir, 'bispectrum', 'test', 'full_vs_bin')

camb_opts = dict(camb_out_dir = opj(ana_dir, 'camb_output/high_acy/nolens'),
                 tag='no_lens',
                 lensed=False)

F = fisher.Fisher()
radii = F.get_updated_radii()
F.get_camb_output(**camb_opts)

bins_full = np.arange(lmin, lmax+1)
bins_binned = F.get_default_bins()

bin_dict = dict(full=bins_full,
                binned=bins_binned)

for typestr in ['full', 'binned']:
    
    bins = bin_dict[typestr]
    F.init_bins(lmin=lmin, lmax=lmax, 
                parity='odd', bins=bins)
    F.init_wig3j()

    bins = F.bins

    #F.beta(radii=radii[::10])
    F.beta(radii=radii, verbose=True)
    F.init_pol_triplets()
    B = F.binned_bispectrum(1, 1, 1)
    B += F.binned_bispectrum(1, -1, -1)
    B += F.binned_bispectrum(-1, 1, 1)
    B += F.binned_bispectrum(-1, -1, -1)
    B += F.binned_bispectrum(+1, -1, +1)
    B += F.binned_bispectrum(-1, +1, -1)



    if F.mpi_rank  == 0:
        print B.shape
        plt.figure()
        plt.imshow(B[0,:,:,0], norm=colors.SymLogNorm(linthresh=1e-24))
        plt.colorbar()
        plt.savefig(opj(out_dir, 'test_{}.png'.format(typestr)))
        plt.close()


        np.save(opj(out_dir, 'test_{}.npy'.format(typestr)), B)
        np.save(opj(out_dir, 'bins_{}.npy'.format(typestr)), F.bins)
        np.save(opj(out_dir, 'num_pass_{}.npy'.format(typestr)), F.num_pass)
        np.save(opj(out_dir, 'first_pass_{}.npy'.format(typestr)), F.first_pass)
        np.save(opj(out_dir, 'beta_s_{}.npy'.format(typestr)), F.depo['scalar']['b_beta'])
        np.save(opj(out_dir, 'beta_t_{}.npy'.format(typestr)), F.depo['tensor']['b_beta'])
        np.save(opj(out_dir, 'pol_trpl_{}.npy'.format(typestr)), F.pol_trpl)
