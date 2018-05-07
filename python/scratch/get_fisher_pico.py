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
sys.path.insert(0,'./../')
import fisher

lmin = 2
lmax = 4000

opj = os.path.join
ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/'
in_dir = opj(ana_dir, 'bispectrum/run_pico')
img_dir = opj(in_dir, 'img')

camb_opts = dict(camb_dir = opj(ana_dir, 'camb_output/high_acy/nolens'),
                 tag='no_lens',
                 lensed=False)

noise_opts = dict(tt_file = opj(ana_dir, 'so_noise/v3', 
            'AdvACT_T_default_Nseasons4.0_NLFyrs2.0_noisecurves_deproj3_mask_16000_ell_TT_yy.txt'),
                  pol_file = opj(ana_dir, 'so_noise/v3', 
            'AdvACT_pol_default_Nseasons4.0_NLFyrs2.0_noisecurves_deproj3_mask_16000_ell_EE_BB.txt')
                  )
F = fisher.Fisher(**camb_opts)
#F.get_noise_curves(cross_noise=False, **noise_opts)
F.init_pol_triplets()

#print F.depo

# Avoid rerunning init_bins(), so load up bins
bins = np.load(opj(in_dir, 'bins.npy'))
F.ells = np.arange(lmin, lmax+1) # bit ugly

# hacky way to update noise
noise_dir = opj(in_dir, 'noise')
nls_pico = np.loadtxt(opj(noise_dir, 'noise_PICO.dat'))
dell = F.ells * (F.ells + 1) / 2. / np.pi
nl_tt = nls_pico[:lmax-1,1]
nl_ee = nls_pico[:lmax-1,15]
nl_bb = nls_pico[:lmax-1,15]

nl_tt = np.ascontiguousarray(nl_tt)
nl_ee = np.ascontiguousarray(nl_ee)
nl_bb = np.ascontiguousarray(nl_bb)

nl_tt /= dell
nl_ee /= dell
nl_bb /= dell

nls = np.ones((6, F.ells.size))
nls[0] = nl_tt 
nls[1] = nl_ee 
nls[2] = nl_bb 
nls[3] = np.zeros_like(nl_bb)
nls[4] = np.zeros_like(nl_bb)
nls[5] = np.zeros_like(nl_bb)
F.depo['nls'] = nls
F.depo['nls_lmin'] = int(lmin)
F.depo['nls_lmax'] = int(lmax)

F.get_binned_invcov(bins=bins)


bin_invcov = F.bin_invcov
bin_cov = F.bin_cov

# plot invcov
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9,7))    
for pidx1, pol1 in enumerate(['T', 'E', 'B']):
    for pidx2, pol2 in enumerate(['T', 'E', 'B']):
        axs[pidx1,pidx2].plot(bins, bin_invcov[:,pidx1,pidx2],
                              label=pol1+pol2)

for ax in axs.reshape(-1):
    ax.legend()
plt.tight_layout()
fig.savefig(opj(img_dir, 'bin_invcov.png'))
plt.close(fig)

# plot cov
dell = bins * (bins + 1) / 2. / np.pi
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9,7))    
for pidx1, pol1 in enumerate(['T', 'E', 'B']):
    for pidx2, pol2 in enumerate(['T', 'E', 'B']):
        axs[pidx1,pidx2].plot(bins, dell * bin_cov[:,pidx1,pidx2],
                              label=pol1+pol2)

for ax in axs.reshape(-1):
    ax.legend()
#    ax.set_xlim(0, 100)
#    ax.set_ylim(0, 0.1)
plt.tight_layout()
fig.savefig(opj(img_dir, 'bin_cov.png'))
plt.close(fig)

# load pol_trpl 
pol_trpl = np.load(opj(in_dir, 'pol_trpl.npy'))

# last bin is not saved in B anymore, so trim
bins = bins[:-1] 

# allocate bin-sized fisher matrix (same size as outer loop)
fisher = np.ones(bins.size) * np.nan

# allocate 12 x 12 cov for use in inner loop
invcov = np.zeros((pol_trpl.size, pol_trpl.size))

# load bispectrum
bispec = np.load(opj(in_dir, 'bispectrum.npy'))
num_pass = np.load(opj(in_dir, 'num_pass.npy'))
num_pass = num_pass[:-1,:-1,:-1]

# create (binned) inverse cov matrix for each ell
# i.e. use the fact that 12x12 pol invcov can be factored
invcov1 = np.ones((bins.size, 12, 12))
invcov2 = np.ones((bins.size, 12, 12))
invcov3 = np.ones((bins.size, 12, 12))

for tidx_a, ptrp_a in enumerate(pol_trpl):
    for tidx_b, ptrp_b in enumerate(pol_trpl):
        # a is first B, b second
        # ptrp = pol triplet

        ptrp_a1 = ptrp_a[0]
        ptrp_a2 = ptrp_a[1]
        ptrp_a3 = ptrp_a[2]

        ptrp_b1 = ptrp_b[0]
        ptrp_b2 = ptrp_b[1]
        ptrp_b3 = ptrp_b[2]

        invcov1[:,tidx_a,tidx_b] = bin_invcov[:-1,ptrp_a1,ptrp_b1]
        invcov2[:,tidx_a,tidx_b] = bin_invcov[:-1,ptrp_a2,ptrp_b2]
        invcov3[:,tidx_a,tidx_b] = bin_invcov[:-1,ptrp_a3,ptrp_b3]

# loop same loop as in binned_bispectrum
for idx1, i1 in enumerate(bins):
    print i1
#    cl1 = bin_invcov[idx1,:]
    cl1 = invcov1[idx1,:,:] # 12x12

    fisher[idx1] = 0.
    
    for idx2, i2 in enumerate(bins[idx1:]):
        idx2 += idx1

#        cl2 = bin_invcov[idx2,:]
        cl2 = invcov2[idx1,:,:] # 12x12

        cl12 = cl1 * cl2

        for idx3, i3 in enumerate(bins[idx2:]):
            idx3 += idx2

            num = num_pass[idx1,idx2,idx3]
            if num == 0:
                continue

#            cl3 = bin_invcov[idx3,:]
            cl123 = cl12 * invcov3[idx3,:,:] #12x12

            # create 12 x 12 inv cov matrix
#            for pidx1 in pol_trpl:
#                for pidx2 in pol_trpl:
                    # pidx are triplets

#                    tmp = cl1[pidx2[0],pidx1[0]]
#                    tmp *= cl2[pidx2[1],pidx1[1]]
#                    tmp *= cl3[pidx2[2],pidx1[2]]
#                    invcov[pidx1,pidx2] = tmp
                    

            B = bispec[idx1,idx2,idx3,:]

            f = np.einsum("i,ij,j", B, cl123, B)
            
            # multiply with delta because each B has 1/delta
            if i1 == i2 == i3:
                f *= 6.
            elif i1 != i2 != i3:
                pass
            else:
                f *= 2.

            # both B's have num 
            f /= float(num**2)

            
            fisher[idx1] += f

min_f = []

for lidx, lmin in enumerate(range(2, 40)):
    f = np.sum(fisher[lmin-2:])
#    print 1/np.sqrt(f)
    print np.sqrt(f)
    min_f.append(np.sqrt(f))

#fisher = np.sum(fisher)
#sigma = 1/np.sqrt(fisher)
#print sigma
min_f = np.array(min_f)
min_f /= min_f[0]

fig, ax = plt.subplots(1,1)
#plt.semilogy(np.arange(2, 200), min_f)
ax.plot(np.arange(2, 40), min_f)
ax.text(0.05, 0.05, r'$\langle B T T \rangle$, $\langle B T E \rangle$, $\langle B E E \rangle$, local',
         transform=ax.transAxes)
#plt.ylim(0,0.1)
ax.set_xlabel(r'$\ell_{\mathrm{min}}$')
ax.set_ylabel(r'$\sigma^{-1} (\sqrt{r} f^{h\zeta\zeta}_{\mathrm{NL}}) \, / \, \sigma^{-1} (\sqrt{r} f^{h\zeta\zeta}_{\mathrm{NL}}, \ell_{\mathrm{min}}=2)$')
fig.savefig(opj(img_dir, 'min_f.png'))
plt.close(fig)

np.savetxt(opj(img_dir, 'f_ell_min.txt'), min_f)

sigma = 1 / np.sqrt(np.cumsum(fisher))

fig, ax = plt.subplots(1,1)
ax.plot(sigma)
fig.savefig(opj(img_dir, 'sigma.png'))
plt.close(fig)



