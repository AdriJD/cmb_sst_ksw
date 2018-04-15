'''
Compare my expression to CAMB output
'''
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import numpy as np

opj = os.path.join

ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/beta/'
ha_beta_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/high_acy/beta_full'

scalar_amp = 2.1e-9
sttr = 0.03
#local_amp = (2 * np.pi)**3 * 16 * np.pi**4 * scalar_amp * np.sqrt(sttr)
local_amp = (2 * np.pi)**3 * scalar_amp * np.sqrt(sttr) * np.sqrt(2) * np.pi # sqrt(2)pi weird, factor (3/5)^2 perhaps?   (5/3)**3 is that factor


#beta_me = np.load(opj(ana_dir, 'beta3.npy')) #ell, L, nfact, err, ks, pol
#err_me = np.load(opj(ana_dir, 'radii3.npy')) #ell, L, nfact, err, ks, pol
beta_me = np.load(opj(ha_beta_dir, 'beta_unique.npy')) #ell, L, nfact, err, ks, pol
err_me = np.load(opj(ha_beta_dir, 'radii_nolens.npy'))
alpha_me = beta_me[:,0,0,:,1,0] #ell, err
beta_me = beta_me[:,0,0,:,0,0] #ell, err


beta_me /= local_amp
#alpha_me /= local_amp
lmax_me = beta_me.shape[0] + 1

alpha = np.load(opj(ana_dir, 'test__alpha.npy'))
beta = np.load(opj(ana_dir, 'test__beta.npy'))
err = np.load(opj(ana_dir, 'test__alpha_beta_r.npy'))

# plot 
step = 100
ell = np.arange(2, alpha.shape[1]+2)
dell = ell * (ell + 1) / 2. / np.pi
fig, ax = plt.subplots(2)
for ridx, rad in enumerate(err[::step]):
    idx = step * ridx
    ax[0].plot(ell, dell * alpha[idx, :])
    ax[1].plot(ell, dell * beta[idx, :])
fig.savefig(opj(ana_dir, 'alpha_beta_dell.png'))
plt.close()

fig, ax = plt.subplots(2)
for lidx, ell in enumerate(ell[::step]):
    if lidx == 0:
        continue
    idx = step * lidx
    ax[0].plot(err, alpha[:, idx], label='{}'.format(ell))
    ax[1].plot(err, beta[:, idx])
    ax[0].legend()
fig.savefig(opj(ana_dir, 'alpha_beta_err.png'))
plt.close()

ell = np.arange(lmax_me+1)
fig, ax = plt.subplots(2, sharex=True)
#for lidx, ell in enumerate(ell):
for ell in [30, 60, 90]:
    ax[0].plot(err_me, beta_me[ell-2, :], label='ell={}'.format(ell))
    ax[1].plot(err, beta[:, ell-2])
ax[0].legend()
fig.savefig(opj(ana_dir, 'beta_comp_err.png'))
plt.close()

fig, ax = plt.subplots(2, sharex=True)
#for lidx, ell in enumerate(ell):
for ell in [30, 60, 90]:
    ax[0].plot(err_me, alpha_me[ell-2, :], label='ell={}'.format(ell))
    ax[1].plot(err, alpha[:, ell-2])
ax[0].legend()
ax [0].set_xlim([12500, 15000])
fig.savefig(opj(ana_dir, 'alpha_comp_err.png'))
plt.close()

print alpha[:, ell-2][0:100]
print alpha_me[ell-2,:][0:100]
print alpha_me.shape
