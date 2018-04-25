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

opj = os.path.join
ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/'

camb_opts = dict(camb_dir = opj(ana_dir, 'camb_output/high_acy/nolens'),
                 tag='no_lens',
                 lensed=False)

noise_opts = dict(tt_file = opj(ana_dir, 'so_noise/v3', 
            'AdvACT_T_default_Nseasons4.0_NLFyrs2.0_noisecurves_deproj3_mask_16000_ell_TT_yy.txt'),
                  pol_file = opj(ana_dir, 'so_noise/v3', 
            'AdvACT_pol_default_Nseasons4.0_NLFyrs2.0_noisecurves_deproj3_mask_16000_ell_EE_BB.txt')
                  )

F = fisher.Fisher(**camb_opts)
F.get_noise_curves(cross_noise=False, **noise_opts)

#print F.get_Ls(279, 317, 319, 'sst')

#exit()
F.init_bins(parity='odd')
F.init_wig3j()

bins = F.bins


radii = F.get_updated_radii()
#F.beta(radii=radii[::10])
F.beta(radii=radii)
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
    plt.savefig(opj(ana_dir, 'bispectrum/test', 'test.png'))
    plt.close()
                
    
    np.save(opj(ana_dir, 'bispectrum/test', 'test.npy'), B)
    np.save(opj(ana_dir, 'bispectrum/test', 'bins.npy'), F.bins)
    np.save(opj(ana_dir, 'bispectrum/test', 'num_pass.npy'), F.num_pass)
    np.save(opj(ana_dir, 'bispectrum/test', 'first_pass.npy'), F.first_pass)
    np.save(opj(ana_dir, 'bispectrum/test', 'beta_s.npy'), F.depo['scalar']['b_beta'])
    np.save(opj(ana_dir, 'bispectrum/test', 'beta_t.npy'), F.depo['tensor']['b_beta'])
    np.save(opj(ana_dir, 'bispectrum/test', 'pol_trpl.npy'), F.pol_trpl)
