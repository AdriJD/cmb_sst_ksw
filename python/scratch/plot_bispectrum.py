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
from scipy.interpolate import interp2d, griddata
sys.path.insert(0,'./../')
import fisher

opj = os.path.join
ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/'

test_dir = opj(ana_dir, 'bispectrum/test')

B = np.load(opj(test_dir, 'test.npy'))
bins = np.load(opj(test_dir, 'bins.npy'))
beta_s = np.load(opj(test_dir, 'beta_s.npy'))
beta_t = np.load(opj(test_dir, 'beta_t.npy'))
pol_trpl = np.load(opj(test_dir, 'pol_trpl.npy'))

idx = 20
lmax = bins[-1]

# ell grid
ell = np.arange(lmax+1)

# data coordinate array

fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(12.,12),
                         sharex=True, sharey=True)

for pidx, pol in enumerate(pol_trpl):

    b1, b2 = np.meshgrid(bins, bins)

    # scale bispectrum by ell**4
    B_slice = B[idx,:,:,pidx]
    B_slice *= (b1 * b2)**2

    # make B symmetric
    B_slice += B_slice.T

    B_slice = B_slice.ravel()
    b1 = b1.ravel()
    b2 = b2.ravel()

    if not np.any(B_slice):
        continue

    # remove zero points
    B_slice_t = B_slice[B_slice != 0.]

    b1 = b1[B_slice != 0.]
    b2 = b2[B_slice != 0.]

    zi = griddata((b1, b2), B_slice_t, (ell[None,:], ell[:,None]), method='nearest',
                  fill_value=np.nan, rescale=True)

    axs_idx = np.divmod(pidx,3)
    ax = axs[axs_idx]
    im = ax.imshow(zi.T, extent=(0,lmax+40,0,lmax+40), 
              origin='lower')#, norm=colors.SymLogNorm(linthresh=1e-24))
    ax.text(0.1, 0.9, str(pol), transform=ax.transAxes)
    fig.colorbar(im, ax=ax)

    # plot 1d slices
    


plt.tight_layout()
plt.savefig(opj(ana_dir, 'bispectrum/test', 'griddata.png'))
plt.close()

