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

ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/'
out_dir = opj(ana_dir, 'camb_output/beta/transfer')
camb_dir = opj(ana_dir, 'camb_output/high_acy/nolens_5200/')
camb_dir_2 = opj(ana_dir, 'camb_output/high_acy/nolens/')

F = fisher.Fisher()
F.get_camb_output(camb_out_dir=camb_dir, tag='no_lens', lensed=False)

tr_si = F.depo['scalar']['transfer']
tr_ti = F.depo['tensor']['transfer']
lmax_si = F.depo['scalar']['lmax']
lmax_ti = F.depo['tensor']['lmax']
k_si = F.depo['scalar']['k']
k_ti = F.depo['tensor']['k']

assert np.array_equal(k_si, k_ti)
assert lmax_si == lmax_ti

# load up transfer fuctions that do not need to be interpolated
F.get_camb_output(camb_out_dir=camb_dir_2, tag='no_lens', lensed=False)

tr_s = F.depo['scalar']['transfer']
tr_t = F.depo['tensor']['transfer']
lmax_s = F.depo['scalar']['lmax']
lmax_t = F.depo['tensor']['lmax']
k_s = F.depo['scalar']['k']
k_t = F.depo['tensor']['k']

assert np.array_equal(k_s, k_t)
assert lmax_s == lmax_t

# plot some k-slices through both transfer functions
ells_i = np.arange(2, lmax_si+1)
ells = np.arange(2, lmax_s+1)

fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True)
#for kidx in [0, 10, 200, 500, 1500, 10000]:
for kidx in [10000]:
    axs[0].plot(ells_i, tr_si[0,:,kidx], label=str(k_si[kidx]))
    
    # find corresponding k in full
    kidx_2 = np.where(k_s >= k_si[kidx])[0][0]
    axs[1].plot(ells, tr_s[0,:,kidx_2], label=str(k_s[kidx_2]))

axs[0].legend()
axs[1].legend()
#axs[1].set_xlim(0, 1000)
fig.savefig(opj(out_dir, 'interp_vs_full.png'))
plt.close(fig)

exit()

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
