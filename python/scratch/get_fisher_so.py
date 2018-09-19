'''
Test binned vs unbinned fisher estimate
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
import camb_tools as ct
import copy

opj = os.path.join

def get_cls(cls_path, lmax, A_lens=1):
    '''
    returns
    -------
    cls : array-like
        Lensed Cls (shape (4,lmax-1) with BB lensing power 
        reduced depending on A_lens. 
        order: TT, EE, BB, TE
    '''
    
    cls_nolens, _ = ct.get_spectra(cls_path, tag='r0',
                             lensed=False, prim_type='tot')
    cls_lensed, _ = ct.get_spectra(cls_path, tag='r0',
                             lensed=True, prim_type='tot')

    # truncate to lmax
    cls_nolens = cls_nolens[:,:lmax-1]
    cls_lensed = cls_lensed[:,:lmax-1]

    BB_nolens = cls_nolens[2]
    BB_lensed = cls_lensed[2]
    
    # difference BB (lensed - unlensed = lens_contribution)
    BB_lens_contr = BB_lensed - BB_nolens

    # depending on A_lens, remove lensing contribution
    cls_lensed[2] -= (1. - A_lens) * BB_lens_contr

    return cls_lensed

def get_nls(lat_path, sac_path, lmax, noise_level_lat='threshold', 
            noise_level_sac='threshold', ell_knee='pessimistic',
            deproj_level=0, fsky=0.1):
    '''

    Arguments
    -----------------
    lat_path : str
        Path to folder containing LAT noise cuves
    sac_path : str
        Path to folder containing SAC noise cuves
    lmax : int
    
    Keyword Arguments
    -----------------
    noise_level_lat : str
        Either 'threshold', 'baseline', 'goal'
    noise_level_sac : str
        Either 'threshold', 'baseline', 'goal'
    ell_knee : str
        Either 'pessimistic' or 'optimistic'
    deproj_level : int
        Foreground cleaning assumption, 0 - 4
        0 is most optimistic
    fsky : float
        Either 0.1, 0.2, 0.4

    Returns
    -------
    nls : array-like
        Shape (6, lmax - 1), order: TT, EE, BB, TE, TB, EB

    Notes
    -----
    Looks like SAC noise curves are only for pol, so use
    SAT TT for TT.       
    '''

    if noise_level_lat == 'threshold':
        sens_lat = 0
    elif noise_level_lat == 'baseline':
        sens_lat = 1
    elif noise_level_lat == 'goal':
        sens_lat = 2

    if noise_level_sac == 'threshold':
        sens_sac = 0
    elif noise_level_sac == 'baseline':
        sens_sac = 1
    elif noise_level_sac == 'goal':
        sens_sac = 2

    if fsky == 0.1:
        fsky_str = '04000'
    elif fsky == 0.2:
        fsky_str = '08000'
    elif fsky == 0.4:
        fsky_str = '16000'

    if ell_knee == 'pessimistic':
        ell_knee_idx = 0
    elif ell_knee == 'optimistic':
        ell_knee_idx = 1


    # init noise curves (fill with 1K^2 noise)
    # truncate later
    nls = np.ones((6, 20000)) * 1e12

    # load up LAT and SAC
    lat_tt_file = 'SOV3_T_default1-4-2_noisecurves_'\
        'deproj{}_SENS{}_mask_{}_ell_TT_yy.txt'.format(deproj_level,
                                                       sens_lat, 
                                                       fsky_str)
    lat_pol_file = lat_tt_file.replace('_T_', '_pol_')
    lat_pol_file = lat_pol_file.replace('_TT_yy', '_EE_BB')

    lat_tt_file = opj(lat_path, lat_tt_file)
    lat_pol_file = opj(lat_path, lat_pol_file)

    sac_file = 'Nl_SO_post_comp_sep_sensitivity_mode'\
        '_{}_one_over_f_mode_{}_TLF_1.5.txt'.format(sens_lat,
                                                   ell_knee_idx)

    sac_file = opj(sac_path, sac_file)
    
    # load
    ells_tt, nl_tt, ells_pol, nl_ee, nl_bb, nl_sac, ells_sac = ct.get_so_noise(
        tt_file=lat_tt_file, pol_file=lat_pol_file, sat_file=sac_file)

    lmin_tt = int(ells_tt[0])
    lmax_tt = int( ells_tt[-1])

    lmin_pol = int( ells_pol[0])
    lmax_pol = int( ells_pol[-1])

    lmin_sac = 2
    lmax_sac = int( ells_sac[-1])

    print lmax_sac

    # combine, first lat then sac
    nls[0,lmin_tt - 2:lmax_tt - 1] = nl_tt 
    nls[1,lmin_pol - 2:lmax_pol - 1] = nl_ee 
    nls[1,lmin_sac - 2:lmax_sac - 1] = nl_sac
    nls[2,lmin_pol - 2:lmax_pol - 1] = nl_bb 
    nls[2,lmin_sac - 2:lmax_sac - 1] = nl_sac
    nls[3] *= 0.
    nls[4] *= 0.
    nls[5] *= 0.
    
    # trunacte to lmax
    nls = nls[:,:lmax - 1]

    return nls

def get_totcov(cls, nls):

    totcov = nls.copy()
    totcov[:4,:] += cls

    return totcov

def get_prim_amp(prim_template='local', scalar_amp=2.1e-9):
    
    common_amp =  16 * np.pi**4 * scalar_amp**2

    if prim_template == 'local':
        return 2 * common_amp

    elif prim_template == 'equilateral':
        return 6 * common_amp

    elif prim_template == 'orthogonal':
        return 6 * common_amp

def load_bispectrum(path, amp):

    bispec = np.load(opj(path, 'bispectrum.npy'))
    bispec[np.isnan(bispec)] = 0.

    bispec *= amp

    return bispec

def load_aux(path):
    '''
    Load bins, num_pass, pol_trpl
    '''

    num_pass = np.load(opj(path, 'num_pass.npy'))

    bins = np.load(opj(path, 'bins.npy'))
    assert np.array_equal(bins, np.unique(bins))

    pol_trpl = np.load(opj(path, 'pol_trpl.npy'))

    return bins, num_pass, pol_trpl

def calc_fisher(lmin, lmax, out_dir, nls_tot, bispec,
                bins, num_pass, pol_trpl,
                prim_template='local',
                fsky=1, A_lens=1., plot_label=''):
#    lmin = 2
#    lmax = 5000


#    ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/'
#    in_dir = opj(ana_dir, 'bispectrum/run_so/')
#    img_dir = opj(in_dir, 'fisher')
#    noise_dir = opj(ana_dir, 'bispectrum/run_pico', 'noise')
#    camb_opts = dict(camb_out_dir = opj(ana_dir, 'camb_output/high_acy/sparse_5000'),
#                     tag='r0',
#                     lensed=False)

#    scalar_amp = 2.1e-9
#    amp =  16 * np.pi**4 * scalar_amp**2

    F = fisher.Fisher()

    lmax = nls_tot.shape[1] + 1
    ells = np.arange(2, lmax+1)

    F.get_binned_invcov(bins=bins, ells=ells, nls_tot=nls_tot)
    bin_invcov = F.bin_invcov
    bin_cov = F.bin_cov

    # plot invcov
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9,7))    
    fig2, axs2 = plt.subplots(nrows=3, ncols=3, figsize=(9,7))    
    for pidx1, pol1 in enumerate(['T', 'E', 'B']):
        for pidx2, pol2 in enumerate(['T', 'E', 'B']):
            axs[pidx1,pidx2].plot(bins, bin_invcov[:,pidx1,pidx2],
                                  label=pol1+pol2)
            axs2[pidx1,pidx2].semilogy(bins, np.abs(bin_invcov[:,pidx1,pidx2]),
                                  label=pol1+pol2)


    for ax in axs.reshape(-1):
        ax.legend()
    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig(opj(out_dir, 'bin_invcov_{}.png'.format(plot_label)))
    fig2.savefig(opj(out_dir, 'bin_invcov_log_{}.png'.format(plot_label)))
    plt.close(fig)
    plt.close(fig2)

    # plot cov
    dell = bins * (bins + 1) / 2. / np.pi
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9,7))    
    fig2, axs2 = plt.subplots(nrows=3, ncols=3, figsize=(9,7))    
    for pidx1, pol1 in enumerate(['T', 'E', 'B']):
        for pidx2, pol2 in enumerate(['T', 'E', 'B']):
            axs[pidx1,pidx2].plot(bins, dell * bin_cov[:,pidx1,pidx2],
                                  label=pol1+pol2)
            axs2[pidx1,pidx2].semilogy(bins, np.abs(dell * bin_cov[:,pidx1,pidx2]),
                                  label=pol1+pol2)


    for ax in axs.reshape(-1):
        ax.legend()
    #    ax.set_xlim(0, 100)
    #    ax.set_ylim(0, 0.1)
    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig(opj(out_dir, 'bin_cov_{}.png'.format(plot_label)))
    fig2.savefig(opj(out_dir, 'bin_cov_log_{}.png'.format(plot_label)))
    plt.close(fig)
    plt.close(fig2)

    # allocate bin-sized fisher matrix (same size as outer loop)
    fisher_per_bin = np.ones(bins.size) * np.nan

    # allocate 12 x 12 cov for use in inner loop
    invcov = np.zeros((pol_trpl.size, pol_trpl.size))

    # create (binned) inverse cov matrix for each ell
    # i.e. use the fact that 12x12 pol invcov can be factored
    # as (Cl-1)_l1^ip (Cl-1)_l2^jq (Cl-1)_l3^kr 
    invcov1 = np.ones((bins.size, 12, 12))
    invcov2 = np.ones((bins.size, 12, 12))
    invcov3 = np.ones((bins.size, 12, 12))

    f_check = 0

    for tidx_a, ptrp_a in enumerate(pol_trpl):
        # ptrp_a = ijk
        for tidx_b, ptrp_b in enumerate(pol_trpl):
            # ptrp_a = pqr
            # a is first bispectrum, b second one
            # ptrp = pol triplet

            ptrp_a1 = ptrp_a[0]
            ptrp_a2 = ptrp_a[1]
            ptrp_a3 = ptrp_a[2]

            ptrp_b1 = ptrp_b[0]
            ptrp_b2 = ptrp_b[1]
            ptrp_b3 = ptrp_b[2]

            invcov1[:,tidx_a,tidx_b] = bin_invcov[:,ptrp_a1,ptrp_b1]
            invcov2[:,tidx_a,tidx_b] = bin_invcov[:,ptrp_a2,ptrp_b2]
            invcov3[:,tidx_a,tidx_b] = bin_invcov[:,ptrp_a3,ptrp_b3]

    # depending on lmin, start outer loop not at first bin
    start_bidx = np.where(bins >= lmin)[0][0]

    # loop same loop as in binned_bispectrum
    for idx1, i1 in enumerate(bins[start_bidx:]):
        idx1 += start_bidx
#        print 'fisher', i1
        cl1 = invcov1[idx1,:,:] # 12x12

        # init
        fisher_per_bin[idx1] = 0.

        for idx2, i2 in enumerate(bins[idx1:]):
            idx2 += idx1
            cl2 = invcov2[idx1,:,:] # 12x12

            cl12 = cl1 * cl2

            for idx3, i3 in enumerate(bins[idx2:]):
                idx3 += idx2

                num = num_pass[idx1,idx2,idx3]
                if num == 0:
                    continue

                cl123 = cl12 * invcov3[idx3,:,:] #12x12

                B = bispec[idx1,idx2,idx3,:]

                f = np.einsum("i,ij,j", B, cl123, B)
                f0 = np.einsum("i,i", B, B)
                b0 = np.einsum("ij,ij", cl123, cl123)

                # both B's have num 
                f /= float(num)

                if i1 == i2 == i3:
                    f /= 6.
                elif i1 != i2 != i3:
                    pass
                else:
                    f /= 2.

                fisher_per_bin[idx1] += f
                f_check += f

    fisher_per_bin *= fsky
    f_check *= fsky
                

    min_f = []

#    print fisher_per_bin
#    print 'fisher:', np.sum(fisher_per_bin)
    print prim_template
    print 'fisher_check:', f_check * (4*np.pi / np.sqrt(8))**2
    print 'sigma:', 1/np.sqrt(f_check) * (np.sqrt(8)/4./np.pi)

    for lidx, lmin in enumerate(range(2, 40)):
        f = np.sum(fisher_per_bin[lmin-2:])
#        print np.sqrt(f)
        min_f.append(np.sqrt(f))

    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(2, 40), min_f)
    ax.text(0.05, 0.05,
            r'$\langle B T T \rangle$, $\langle B T E \rangle$, $\langle B E E \rangle$, local',
             transform=ax.transAxes)
    #plt.ylim(0,0.1)
    ax.set_xlabel(r'$\ell_{\mathrm{min}}$')
    ax.set_ylabel(r'$\sigma^{-1} (\sqrt{r} f^{h\zeta\zeta}_{\mathrm{NL}}) \, / \, \sigma^{-1} (\sqrt{r} f^{h\zeta\zeta}_{\mathrm{NL}}, \ell_{\mathrm{min}}=2)$')
    fig.savefig(opj(out_dir, 'min_f_{}.png'.format(plot_label)))
    plt.close(fig)

    sigma = 1 / np.sqrt(np.cumsum(fisher_per_bin))

    fig, ax = plt.subplots(1,1)
    ax.plot(bins, sigma)
    fig.savefig(opj(out_dir, 'sigma_{}.png'.format(plot_label)))
    plt.close(fig)

    fig, ax = plt.subplots(1,1)
    ax.semilogy(bins, fisher_per_bin)
    fig.savefig(opj(out_dir, 'fisher_ell_{}.png'.format(plot_label)))
    plt.close(fig)

if __name__ == '__main__':

    ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/'
    cls_path = opj(ana_dir, 'camb_output/high_acy/sparse_5000')
    lat_path = opj(ana_dir, 'so_noise/v3/so')
    sac_path = opj(ana_dir, 'so_noise/sat')
    lmin = 30 
    lmax = 5000
    fsky = 0.1
    bispec_path = dict(local=opj(ana_dir, 'bispectrum/run_so/local'),
                       equilateral=opj(ana_dir, 'bispectrum/run_so/equilateral'),
                       orthogonal=opj(ana_dir, 'bispectrum/run_so/orthogonal'))

    threshold_opts = dict(noise_level_lat='threshold',
                          noise_level_sac='threshold',
                          ell_knee='pessimistic',
                          A_lens=1.) 

    baseline_opts = dict(noise_level_lat='baseline',
                          noise_level_sac='baseline',
                          ell_knee='pessimistic',
                          A_lens=0.75)

    goal_opts = dict(noise_level_lat='goal',
                          noise_level_sac='goal',
                          ell_knee='optimistic',
                          A_lens=0.5)

    for prim_template in ['local', 'equilateral', 'orthogonal']:
#    for prim_template in ['orthogonal']:
#    for prim_template in ['equilateral']:
        # load bispectrum
        b_path = bispec_path[prim_template]
        fisher_path = opj(b_path, 'fisher')

        amp = get_prim_amp(prim_template=prim_template)

        bispec = load_bispectrum(b_path, amp)
        bins, num_pass, pol_trpl = load_aux(b_path)

        for noise_opts, n_label in zip([threshold_opts, baseline_opts, goal_opts],
                              ['threshold', 'baseline', 'goal']):
#        for noise_opts in [threshold_opts]:#, baseline_opts, goal_opts]:                

            noise_opts = copy.deepcopy(noise_opts)
            A_lens = noise_opts.pop('A_lens')
            # load Cls
            cls = get_cls(cls_path, lmax, A_lens=A_lens)
                
            for deproj in [0,3]:
#            for deproj in [0]:
                # load noise
                nls = get_nls(lat_path, sac_path, lmax, deproj_level=deproj, 
                              fsky=fsky, **noise_opts)

                totcov = get_totcov(cls, nls)                

                plot_label  = '{}_{}_{}'.format(prim_template, n_label, deproj)
                                       
                print plot_label                       
                calc_fisher(lmin, lmax, fisher_path, totcov, bispec,
                            bins, num_pass, pol_trpl,
                            prim_template=prim_template,
                            fsky=fsky, A_lens=A_lens,
                            plot_label=plot_label)


