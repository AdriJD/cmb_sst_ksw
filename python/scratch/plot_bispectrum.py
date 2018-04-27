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
num_pass = np.load(opj(test_dir, 'num_pass.npy'))
num_pass = num_pass[:-1,:-1,:-1]

# For plotting, divide out num pass, note that last bin is removed from bispectrum
B[num_pass != 0,:] /= num_pass[num_pass != 0,np.newaxis]

bins = np.load(opj(test_dir, 'bins.npy'))
bins = bins[:-1] # last bin is not saved in B anymore
#beta_s = np.load(opj(test_dir, 'beta_s.npy'))
#beta_t = np.load(opj(test_dir, 'beta_t.npy'))
pol_trpl = np.load(opj(test_dir, 'pol_trpl.npy'))

idx = 70
lmin = bins[idx]
lmax = bins[-1]

# ell grid
ell = np.arange(lmax+1)

# interpolated 
fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(12.,12),
                         sharex=True, sharey=True)
# raw
fig_r, axs_r = plt.subplots(nrows=4, ncols=3, figsize=(12.,12),
                          sharex=True, sharey=True)

# raw scatter
fig_s, axs_s = plt.subplots(nrows=4, ncols=3, figsize=(12.,12),
                          sharex=True, sharey=True)

# 1d raw 
fig_d, axs_d = plt.subplots(nrows=4, ncols=3, figsize=(12.,12),
                          sharex=True, sharey=False)

for pidx, pol in enumerate(pol_trpl):

    axs_idx = np.divmod(pidx,3)
    ax = axs[axs_idx]
    ax_r = axs_r[axs_idx]
    ax_s = axs_s[axs_idx]
    ax_d = axs_d[axs_idx]

    b1, b2 = np.meshgrid(bins, bins)

    # scale bispectrum by ell**4
    B_slice = B[idx,:,:,pidx]
    B_slice *= (b1 * b2)**2

    # make B antisymmetric
    B_slice += -B_slice.T
    
    B_slice_r = B_slice.copy()
    B_slice_r[B_slice_r == 0] = np.nan

    im_r = ax_r.imshow(np.abs(B_slice_r.T), extent=(lmin,lmax,lmin,lmax), 
                       origin='lower')#, norm=colors.SymLogNorm(linthresh=1e-24))

    Bl = np.diagonal(B_slice_r, offset=2)
    im_d = ax_d.plot(np.abs(Bl))

    B_slice = B_slice.ravel()
    b1 = b1.ravel()
    b2 = b2.ravel()

    im_s = ax_s.scatter(b1, b2, s=0.4, marker='s', c=np.abs(B_slice_r.ravel()),
                        cmap='viridis')

    # remove zero points
    B_slice_t = B_slice[B_slice != 0.]

    b1 = b1[B_slice != 0.]
    b2 = b2[B_slice != 0.]

    zi = griddata((b1, b2), np.abs(B_slice_t), (ell[None,:], ell[:,None]), method='nearest',
                  fill_value=np.nan, rescale=True)

    im = ax.imshow(np.abs(zi.T), extent=(0,lmax,0,lmax), 
              origin='lower')#, norm=colors.SymLogNorm(linthresh=1e-24))
    ax.text(0.1, 0.9, str(pol), transform=ax.transAxes)
    ax_r.text(0.1, 0.9, str(pol), transform=ax_r.transAxes)
    ax_s.text(0.1, 0.9, str(pol), transform=ax_s.transAxes)
    ax_d.text(0.1, 0.9, str(pol), transform=ax_d.transAxes)

    ax_s.set_xlim(0, lmax)
    ax_s.set_ylim(0, lmax)

    fig.colorbar(im, ax=ax)
    fig_r.colorbar(im_r, ax=ax_r)
    fig_s.colorbar(im_s, ax=ax_s)

#plt.tight_layout()
fig.tight_layout()
fig_r.tight_layout()
fig_s.tight_layout()
fig_d.tight_layout()

fig.savefig(opj(ana_dir, 'bispectrum/test', 'griddata.png'))
fig_r.savefig(opj(ana_dir, 'bispectrum/test', 'griddata_raw.png'))
fig_s.savefig(opj(ana_dir, 'bispectrum/test', 'griddata_raw_scatter.png'))
fig_d.savefig(opj(ana_dir, 'bispectrum/test', 'griddata_1d.png'))
plt.close(fig)
plt.close(fig_r)
plt.close(fig_s)
plt.close(fig_d)

