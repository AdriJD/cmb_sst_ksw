'''

'''

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
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
F.init_bins()
F.init_wig3j()

print F.first_pass.shape
print F.num_pass.shape
bins = F.bins

#for i in xrange(bins.size-1):
#    for j in xrange(bins.size-1):
#        for k in xrange(bins.size-1):

#for idx1, i1 in enumerate(bins[:-1]):
#    for idx2, i2 in enumerate(bins[idx1:-1]):
#        idx2 += idx1
#        for idx3, i3 in enumerate(bins[idx2:-1]):
#            idx3 += idx2
#            print idx1,idx2,idx2, F.num_pass[idx1,idx2,idx3], F.first_pass[idx1,idx2,idx3,:]


#            print i,j,k, F.num_pass[i,j,k], F.first_pass[i,j,k,:]

#print F.wig_s
#print F.wig_t

#exit()

radii = F.get_updated_radii()
F.beta(radii=radii[::10])
F.init_pol_triplets()
if F.mpi_rank == 0:
    F.binned_bispectrum(-1, 0, 1)

