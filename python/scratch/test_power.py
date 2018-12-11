'''
Use transfer function to calculate Cl.
'''

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import trapz
sys.path.insert(0,'./../')
from sst import PreCalc

opj = os.path.join

ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20180911_sst/'
ana_dir2 = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/'
#out_dir = opj(ana_dir, 'transfer2cl')
#out_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20181128_sst_py_vs_fortran/img'
out_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20181211_sst_changes/img'
#camb_dir = opj(ana_dir, 'camb_output/lensed_r0_4000/')
#camb_dir = opj(ana_dir2, 'camb_output/high_acy/nolens/high_r') #NOTE high r
camb_dir = opj(ana_dir2, 'camb_output/high_acy/nolens')
#camb_dir = opj(ana_dir2, 'camb_output/high_acy/no_lens_4000')

F = PreCalc()

F.get_camb_output(camb_out_dir=camb_dir, tag='no_lens', lensed=False, prim_type='scalar',
                  high_ell=False)
cls_camb_s = F.depo['cls'].copy()

F.get_camb_output(camb_out_dir=camb_dir, tag='no_lens', lensed=False, prim_type='tensor',
                  high_ell=False)
cls_camb_t = F.depo['cls'].copy()

tr_s = F.depo['scalar']['transfer']
tr_t = F.depo['tensor']['transfer']
lmax_s = F.depo['scalar']['lmax']
lmax_t = F.depo['tensor']['lmax']
lmin = 2

k_s = F.depo['scalar']['k']
k_t = F.depo['tensor']['k']


# plot scalar transfer
fig, axs = plt.subplots(nrows=4, sharex=True)
axs[0].semilogx(k_s, tr_s[0,40,:])
axs[1].semilogx(k_s, tr_s[1,40,:])
axs[2].semilogx(k_s, k_s**(-1))
axs[3].semilogx(k_s, k_s**(-1)*tr_s[1,40,:]**2)
fig.savefig(opj(out_dir, 'tr_k.png'))
plt.tight_layout()
plt.close(fig)

assert np.array_equal(k_s, k_t)
assert lmax_s == lmax_t
lmax = lmax_s

scalar_amp = 2.1e-9  * 2 * np.pi**2 
ns = 0.96
r = 0.03
pivot = 0.05

# create primordial power spectra
P_k_s = k_s**(-3) * (k_s / pivot) ** (ns-1)
P_k_s *= scalar_amp 

P_k_t = k_s**(-3)
#P_k_t *= scalar_amp * r * np.pi ** 2 # as in shiraisi (P_T = r P_S / 2
P_k_t *= scalar_amp * r # as in shiraisi (P_T = r P_S / 2


k2 = k_s**2
P_k_s *= k2
P_k_t *= k2

# only TT scalar
cl_s = np.ones((4, lmax_s - lmin + 1)) * np.nan
cl_t = np.ones((4, lmax_t - lmin + 1)) * np.nan

ells = np.arange(lmin, lmax+1)
dells = ells * (ells + 1) / 2. / np.pi

ells_f = np.arange(lmin, lmax+1)
dells_f = np.arange(lmin, lmax+1)

for lidx, ell in enumerate(ells):
    
    lidx_tr = lidx + (lmin - 2)
    lidx_f = ell - lmin

    cl_s[0,lidx_f] = trapz(P_k_s  * tr_s[0,lidx_tr,:]**2, k_s)    
    cl_t[0,lidx_f] = trapz(P_k_t  * tr_t[0,lidx_tr,:]**2, k_s) 

    # EE
    cl_s[1,lidx_f] = trapz(P_k_s  * tr_s[1,lidx_tr,:]**2, k_s) 
    cl_t[1,lidx_f] = trapz(P_k_t  * tr_t[1,lidx_tr,:]**2, k_s)

    # BB
    cl_t[2,lidx_f] = trapz(P_k_t  * tr_t[2,lidx_tr,:]**2, k_s)

    #TE
    cl_s[3,lidx_f] = trapz(P_k_s * tr_s[0,lidx_tr,:] * tr_s[1,lidx_tr,:], k_s) 
    cl_t[3,lidx_f] = trapz(P_k_t * tr_t[0,lidx_tr,:] * tr_t[1,lidx_tr,:], k_s) 

cl_s *= (2 / np.pi)
cl_t *= (2 / np.pi)

cl_camb_s = cls_camb_s[:,lmin-2:lmax-1]
cl_camb_t = cls_camb_t[:,lmin-2:lmax-1]


# scalar
fig, axs = plt.subplots(nrows=3, ncols=4, sharex=True)
msk = np.isfinite(cl_s[0,:])
axs[0,0].semilogy(ells_f[msk], dells_f[msk] * cl_s[0,:][msk])
axs[1,0].semilogy(ells_f, dells_f * cl_camb_s[0])
axs[2,0].plot(ells_f[msk], (cl_camb_s[0][msk] / cl_s[0,:][msk]))
msk = np.isfinite(cl_s[1,:])
axs[0,1].semilogy(ells_f[msk], dells_f[msk]  * cl_s[1,:][msk])
axs[1,1].semilogy(ells_f, dells_f * cl_camb_s[1,:])
axs[2,1].plot(ells_f[msk], (cl_camb_s[1][msk] / (cl_s[1][msk])))
msk = np.isfinite(cl_s[3,:])
axs[0,3].plot(ells_f[msk], dells_f[msk] * cl_s[3,:][msk])
axs[1,3].plot(ells_f, dells_f * cl_camb_s[2,:]) # scalCl is TT, EE, TE
axs[2,3].plot(ells_f[msk], (cl_camb_s[2][msk] / cl_s[3][msk]))

fig.savefig(opj(out_dir, 'cl_s.png'))
plt.tight_layout()
plt.close(fig)

# tensor
fig, axs = plt.subplots(nrows=3, ncols=4, sharex=True)
msk = np.isfinite(cl_t[0,:])
axs[0,0].semilogy(ells_f[msk], dells_f[msk] * cl_t[0,:][msk])
axs[1,0].semilogy(ells_f, dells_f * cl_camb_t[0])
axs[2,0].plot(ells_f[msk], (cl_camb_t[0][msk] / cl_t[0,:][msk]))
msk = np.isfinite(cl_t[1,:])
axs[0,1].semilogy(ells_f[msk], dells_f[msk]  * cl_t[1,:][msk])
axs[1,1].semilogy(ells_f, dells_f * cl_camb_t[1,:])
axs[2,1].plot(ells_f[msk], (cl_camb_t[1][msk] / (cl_t[1][msk])))
msk = np.isfinite(cl_t[2,:])
axs[0,2].semilogy(ells_f[msk], dells_f[msk] * cl_t[2,:][msk])
axs[1,2].semilogy(ells_f, dells_f * cl_camb_t[2,:])
axs[2,2].plot(ells_f[msk], (cl_camb_t[2][msk] / cl_t[2][msk]))
msk = np.isfinite(cl_t[3,:])
axs[0,3].plot(ells_f[msk], dells_f[msk] * cl_t[3,:][msk])
axs[1,3].plot(ells_f, dells_f * cl_camb_t[3,:])
axs[2,3].plot(ells_f[msk], (cl_camb_t[3][msk] / cl_t[3][msk]))

fig.savefig(opj(out_dir, 'cl_t.png'))
plt.tight_layout()
plt.close(fig)




