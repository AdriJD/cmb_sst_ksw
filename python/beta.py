'''
calculate beta
'''

import sys
import os
import numpy as np
import fisher

opj = os.path.join

#ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/beta/'
ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/high_acy/beta_full/'
F = fisher.Fisher()
camb_dir='/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/high_acy/sparse_5000'
#F = fisher.Fisher(camb_dir='/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/')
#F.get_camb_output(tag='no_lens')
F.get_camb_output(camb_out_dir=camb_dir, tag='r0', lensed=False)

local, amp =  F.local()
radii = F.get_updated_radii()[::10]
F.init_bins(lmin=4000, lmax=5000, verbose=True)
beta = F.beta(f=local, radii=radii, verbose=True)


#if F.mpi_rank == 0:
#    np.save(opj(ana_dir, 'beta_high_ell.npy'), beta)
#    np.save(opj(ana_dir, 'radii_high_ell.npy'), radii)
