'''
Look at combination of full and sparse transfers
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
camb_dir = opj(ana_dir, 'camb_output/high_acy/sparse_5000/')

F = fisher.Fisher()
F.get_camb_output(camb_out_dir=camb_dir, tag='r0', lensed=False, 
                  high_ell=True)

tr_s = F.depo['scalar']['transfer']
tr_t = F.depo['tensor']['transfer']
lmax_s = F.depo['scalar']['lmax']
lmax_t = F.depo['tensor']['lmax']
k_s = F.depo['scalar']['k']
k_t = F.depo['tensor']['k']

ells = F.depo['scalar']['ells_sparse']

assert np.array_equal(k_s, k_t)
assert lmax_s == lmax_t

# load up low ell transfer only
F.get_camb_output(camb_out_dir=camb_dir, tag='r0', lensed=False, 
                  high_ell=False)

tr_s_l = F.depo['scalar']['transfer']
tr_t_l = F.depo['tensor']['transfer']
lmax_s_l = F.depo['scalar']['lmax']
lmax_t_l = F.depo['tensor']['lmax']
k_s_l = F.depo['scalar']['k']
k_t_l = F.depo['tensor']['k']

# plot some ell-slices through both transfer functions
# to check if k-interpolation at high ell went okay
ells_l = np.arange(2, lmax_s_l+1)


#print tr_s[0,:,8000]
#print tr_s[0,:,8000][::10]
#print tr_s_l[0,:,8000][::10]

fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)

for ell in [3500, 4000, 4010, 4500, 5000]:
#for ell in [4010, 4500, 5000]:
    
    lidx = np.where(ell <= ells)[0][0]

    print tr_s[0,lidx,:]
    print np.any(tr_s[0,lidx,:])

    axs[0,0].semilogy(k_s, np.abs(tr_s[0,lidx,:]), label=str(ells[lidx]))
    axs[0,1].semilogy(k_t, np.abs(tr_t[0,lidx,:]), label=str(ells[lidx]))

    try:
        lidx_l = np.where(ell <= ells_l)[0][0]
        axs[1,0].semilogy(k_s_l, np.abs(tr_s_l[0,lidx_l,:]), label=str(ells[lidx_l]))
        axs[1,1].semilogy(k_t_l, np.abs(tr_t_l[0,lidx_l,:]), label=str(ells[lidx_l]))
    except IndexError:
        pass
    



axs[0,0].legend()
axs[0,1].legend()
axs[1,0].legend()
axs[1,1].legend()
axs[1,1].set_xlim(0.2, 1)
fig.tight_layout()
fig.savefig(opj(out_dir, 'interp_high_ell.png'))
plt.close(fig)

exit()

