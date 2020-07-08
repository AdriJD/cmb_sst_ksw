import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from cycler import cycler
import os
import sys
import numpy as np
from scipy.io import FortranFile

opj  = os.path.join

source_dir = '../ancillary/camb_delta_p_l_k/'
#lmax = 2200 #scalar
lmax = 1500 #tensor

# read tensor transfer function
f = FortranFile(opj(source_dir, 'Delta_p_l_k_tensor.dat'), 'r')
delta_p_l_k = f.read_reals(float)
f.close()

# read tensor ells
f = FortranFile(opj(source_dir, 'l_tensor.dat'), 'r')
ell = f.read_reals(np.int32)
f.close()

# truncate ell array (because written file contains lots of appended zeros)
ell = ell[:lmax-1]

# read tensor ks
f = FortranFile(opj(source_dir, 'points_tensor.dat'), 'r')
k = f.read_reals(float)
f.close()

# read tensor num sources
f = FortranFile(opj(source_dir, 'NumSources_tensor.dat'), 'r')
num_sources = f.read_reals(np.int32)
f.close()

# reshape
transfer = delta_p_l_k.reshape((num_sources[0], ell.size, k.size), order='F')


# print tr(el) as func of k
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12,6), sharex=True)
#for idx, ell_idx in enumerate(xrange(4)):
for ax, ell_idx in zip(axs.reshape(-1), [0, 30, 298, 1298]):
    ax.semilogx(k, transfer[0,ell_idx,:])
#    ax.plot(k, transfer[0,ell_idx,:])
    ax.set_xlim([1e-5, 0.4])
    ax.set_xlabel(r'$k \mathrm{Mpc}$')
    ax.set_title(r'$\ell = {}$'.format(ell[ell_idx]))
fig.tight_layout()
fig.suptitle('T transfer functions')
fig.subplots_adjust(top=0.88)
fig.savefig('tranfers.png')

num = k.size
#num = 100
colors = plt.get_cmap('inferno')(np.linspace(0, 1, num))

dell = ell * (ell + 1) / 2. /np.pi

# print tr(k) as func of ell
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6), sharex=True)
ax.set_prop_cycle('color', colors)
for kidx in xrange(num):
#    ax.semilogx(ell, np.abs(transfer[0,:,kidx]))
#    ax.loglog(ell, np.abs(transfer[0,:,kidx]))
#    ax.semilogx(ell, transfer[0,:,kidx])
    ax.plot(ell, transfer[0,:,kidx])
ax.set_yscale('symlog')
ax.set_xscale('log')
ax.set_ylim([-1e-9, 1e-9])
ax.set_xlim([20, lmax])
ax.set_xlabel(r'$\ell$')

fig.tight_layout()
fig.suptitle('T transfer functions')
fig.subplots_adjust(top=0.88)
fig.savefig('tranfers_ell.png')

