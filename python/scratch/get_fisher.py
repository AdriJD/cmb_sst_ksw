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

camb_opts = dict(camb_dir = opj(ana_dir, 'camb_output/high_acy/nolens'),
                 tag='no_lens',
                 lensed=False)

noise_opts = dict(tt_file = opj(ana_dir, 'so_noise/v3', 
            'AdvACT_T_default_Nseasons4.0_NLFyrs2.0_noisecurves_deproj3_mask_16000_ell_TT_yy.txt'),
                  pol_file = opj(ana_dir, 'so_noise/v3', 
            'AdvACT_pol_default_Nseasons4.0_NLFyrs2.0_noisecurves_deproj3_mask_16000_ell_EE_BB.txt')
                  )
F = fisher.Fisher(**camb_opts)
F.get_noise_curves(cross_noise=False, **noise_opts)
F.init_pol_triplets()

#print F.depo

# Avoid rerunning init_bins(), so load up bins
bins = np.load(opj(in_dir, 'bins.npy'))
F.ells = np.arange(lmin, lmax) # bit ugly

F.get_binned_invcov(bins=bins)

print F.bin_invcov.shape
print bins.shape

bin_invcov = F.bin_invcov

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

first_pass = np.load(opj(in_dir, 'first_pass.npy'))
fist_pass = first_pass[:-1,:-1,:-1]

print bispec.shape
print pol_trpl.shape

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

            
            fisher[idx1] += f

print np.any(np.isnan(fisher))
print fisher
sigma = 1 / np.sqrt(np.cumsum(fisher))
#sigma /= sigma[0]

print np.sum(sigma)
print np.sum(sigma[1:])
print np.sum(sigma[2:])
print np.sum(sigma[3:])
#print 1 / np.sqrt(np.cumsum(fisher) / np.cumsum(fisher)[0] )

            
# pol matrix
# weigh by 1/Di1i2i3

# add fisher to fisher array

# end loop

# save fisher

# sum different parts of Fisher array






