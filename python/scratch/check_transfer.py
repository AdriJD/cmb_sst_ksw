'''
Look at transfer functions 
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

ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/beta/transfer'

F = fisher.Fisher(
    camb_dir='/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/high_acy/nolens/')
#F = fisher.Fisher(camb_dir='/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/')
#F.get_camb_output(tag='test')
F.get_camb_output(tag='no_lens', lensed=False)
radii = F.get_updated_radii()

tr = F.depo['tensor']['transfer']
k = F.depo['tensor']['k']
lmax = F.depo['tensor']['lmax']

print tr.shape

print lmax
print k.size
print radii.size

#print k
#print (k * radii[500])[::10]


cls = F.cls
print cls.shape

ell = 100
ridx = 400
print radii[ridx]

fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
axs[0].plot(k, tr[0, ell-2, :], ls=':')
axs[1].plot(k, spherical_jn(ell, k * radii[ridx]), ls=':')
axs[2].scatter(k, k, s=0.1)
#axs[0].set_xlim([0.05, 0.15])
#axs[2].set_ylim([0, 0.05])
plt.savefig(opj(ana_dir, 'ell_{}_rad_{}.png'.format(ell-2, radii[ridx])))
plt.close()
