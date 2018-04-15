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
F = fisher.Fisher(
    camb_dir='/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/high_acy/nolens')
#F = fisher.Fisher(camb_dir='/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/')
#F.get_camb_output(tag='no_lens')
F.get_camb_output(tag='no_lens', lensed=False)

local =  F.local()
radii = F.get_updated_radii()
#beta = F.beta(local, 'scalar', [0], radii[:10])
beta = F.beta(local, 'scalar', radii=radii)

#print local
#exit()
print beta[:10,0,0,:,0,0]
print np.any(np.isnan(beta))


#np.save(opj(ana_dir, 'beta_{}.npy'.format(F.mpi_rank)), beta)
#np.save(opj(ana_dir, 'radii_{}.npy'.format(F.mpi_rank)), F.radii_sub)
if F.mpi_rank == 0:
    np.save(opj(ana_dir, 'beta_nolens.npy'), beta)
    np.save(opj(ana_dir, 'radii_nolens.npy'), radii)
