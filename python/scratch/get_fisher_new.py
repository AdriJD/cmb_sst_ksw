import os
import numpy as np
from sst import Fisher
from sst import camb_tools as ct
from sst import plot_tools

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

def get_nls(lat_path, lmax, sac_path=None,
            deproj_level=0):
    '''

    Arguments
    -----------------
    lat_path : str
        Path to folder containing LAT noise cuves
    lmax : int
    
    Keyword Arguments
    -----------------
    sac_path : str, None
        Path to folder containing SAC noise cuves
    deproj_level : int
        Foreground cleaning assumption, 0 - 4
        0 is most optimistic

    Returns
    -------
    nls : array-like
        Shape (6, lmax - 1), order: TT, EE, BB, TE, TB, EB

    Notes
    -----
    Looks like SAC noise curves are only for pol, so use
    SAT TT for TT.       
    '''

    # Add option to skip SAT.
    # SO V3 (deproj0, S2(goal) 16000 deg2

    # init noise curves (fill with 1K^2 noise)
    # truncate later
    nls = np.ones((6, 20000)) * 1e12

    # load up LAT
#    lat_tt_file = 'S4_2LAT_T_default_noisecurves_'\
#        'deproj{}_SENS0_mask_16000_ell_TT_yy.txt'.format(deproj_level) # NOTE
    lat_tt_file = 'SOV3_T_default1-4-2_noisecurves_'\
        'deproj{}_SENS2_mask_16000_ell_TT_yy.txt'.format(deproj_level)
                                                       
    lat_pol_file = lat_tt_file.replace('_T_', '_pol_')
    lat_pol_file = lat_pol_file.replace('_TT_yy', '_EE_BB')

    lat_tt_file = opj(lat_path, lat_tt_file)
    lat_pol_file = opj(lat_path, lat_pol_file)

    # load lat
    ells_tt, nl_tt, ells_pol, nl_ee, nl_bb = ct.get_so_noise(
        tt_file=lat_tt_file, pol_file=lat_pol_file, sat_file=None)

    lmin_tt = int(ells_tt[0])
    lmax_tt = int(ells_tt[-1])

    #lmin_pol = int(ells_pol[0])
    lmin_pol = 30 # as suggested on wiki
    lmax_pol = int(ells_pol[-1])

    if sac_path is not None:
        sac_file = 'Db_noise_04.00_ilc_bin3_av.dat'
        sac_file = opj(sac_path, sac_file)
    
        # load sac, note these are Dell bandpowers
        ell, sac_ee, sac_bb = np.loadtxt(sac_file).transpose()
        dell = ell * (ell + 1) / 2. / np.pi
        sac_ee /= dell
        sac_bb /= dell

        # interpolate
        lmin_sac = int(ell[0])
        lmax_sac = int(ell[-1])
        ell_f = np.arange(lmin_sac, lmax_sac+1)
        sac_ee = np.interp(ell_f, ell, sac_ee)
        sac_bb = np.interp(ell_f, ell, sac_bb)

    # combine, first lat then (if needed )sac because lat has lower lmin
    nls[0,lmin_tt - 2:lmax_tt - 1] = nl_tt 
    nls[1,lmin_pol - 2:lmax_pol - 1] = nl_ee[ells_pol >= lmin_pol] 
    nls[2,lmin_pol - 2:lmax_pol - 1] = nl_bb[ells_pol >= lmin_pol] 
    nls[3] *= 0.
    nls[4] *= 0.
    nls[5] *= 0.
    
    if sac_path is not None:
        nls[1,lmin_sac - 2:lmax_sac - 1] = sac_ee
        nls[2,lmin_sac - 2:lmax_sac - 1] = sac_bb

    # trunacte to lmax
    nls = nls[:,:lmax - 1]

    return nls

def get_fiducial_nls(noise_amp_temp, noise_amp_pol, lmax):
    '''
    Create N_{\ell} = noise_amp^2 noise arrays.

    Arguments
    -----------------
    noise_amp_temp : float
        Noise ampltidue in uK arcmin.
    noise_amp_pol : float
    lmax : int
    
    Returns
    -------
    nls : array-like
        Shape (6, lmax - 1), order: TT, EE, BB, TE, TB, EB
    '''

    # init noise curves (fill with 1K^2 noise)
    # truncate later
    nls = np.ones((6, 20000)) * 1e12

    # N_{\ell} = uK^2 radians^2
    arcmin2radians = np.pi / 180. / 60.
    noise_amp_temp *= arcmin2radians
    noise_amp_pol *= arcmin2radians

    # combine, first lat then sac because lat has lower lmin
    nls[0,:] = noise_amp_temp ** 2
    nls[1,:] = noise_amp_pol ** 2
    nls[2,:] = noise_amp_pol ** 2 
    nls[3] *= 0.
    nls[4] *= 0.
    nls[5] *= 0.
    
    # trunacte to lmax
    nls = nls[:,:lmax - 1]

    return nls

def get_prim_amp(prim_template='local', scalar_amp=2.1e-9):

    common_amp =  16 * np.pi**4 * scalar_amp**2

    if prim_template == 'local':
        return 2 * common_amp

    elif prim_template == 'equilateral':
        return 6 * common_amp

    elif prim_template == 'orthogonal':
        return 6 * common_amp

def get_totcov(cls, nls, no_ee=False, no_tt=False):

    totcov = nls.copy()
    totcov[:4,:] += cls

    if no_ee:
        totcov[1,:] = 1e12
    if no_tt:
        totcov[0,:] = 1e12

    return totcov

def run_fisher(template, ana_dir, camb_dir, totcov, ells, lmin=2, lmax=4999,fsky=0.03, 
               plot_tag='', tag=None):
    
    F = Fisher(ana_dir)
    camb_opts = dict(camb_out_dir=camb_dir,
                     tag='r0',
                     lensed=False,
                     high_ell=True)

    F.get_camb_output(**camb_opts)
    radii = F.get_updated_radii()
    radii = radii[::2]
    F.get_bins(lmin=lmin, lmax=lmax, load=True, verbose=False, 
               parity='odd', tag=tag+'_new')
 #   F.get_beta(func='equilateral', load=True, verbose=False, radii=radii, tag=tag)
    F.get_beta(func='equilateral', load=True, verbose=True, radii=radii, tag=tag+'_new',
               interp_factor=10)
#    F.get_binned_bispec(template, load=True, tag=tag)
    F.get_binned_bispec(template, load=True, tag=tag+'_new') 
    bin_invcov, bin_cov = F.get_binned_invcov(ells, totcov, return_bin_cov=True)

    # Plot invcov, cov
    plot_opts = dict(lmin=2)
    bins = F.bins['bins']
    plot_tools.cls_matrix(plot_tag, bins, bin_invcov, log=False, plot_dell=False,
                          inv=True, **plot_opts)
    plot_tools.cls_matrix(plot_tag.replace('invcov', 'cov_dell'),
                          bins, bin_cov, log=False, plot_dell=True,
                          **plot_opts)
    print(lmin, lmax)
    fisher = F.naive_fisher(bin_invcov, lmin=lmin, lmax=lmax, fsky=fsky)
    sigma = 1/np.sqrt(fisher)
    
    return fisher, sigma

    ###### OLD

    amp = get_prim_amp(template)
    F.bispec['bispec'] *= amp

    F.get_binned_invcov(nls=totcov)
    bin_invcov = F.bin_invcov
    bin_cov = F.bin_cov

    bin_size = F.bins['bins'].size
    bins = F.bins['bins']
    num_pass = F.bins['num_pass_full']
    bispec = F.bispec['bispec']

    # Plot invcov, cov
    plot_opts = dict(lmin=2)
    plot_tools.cls_matrix(plot_tag, bins, bin_invcov, log=False, plot_dell=False,
                          inv=True, **plot_opts)
    plot_tools.cls_matrix(plot_tag.replace('invcov', 'cov_dell'),
                          bins, bin_cov, log=False, plot_dell=True,
                          **plot_opts)
    plot_tools.cls_matrix(plot_tag.replace('invcov', 'cov'),
                          bins, bin_cov, log=False, plot_dell=False, 
                          **plot_opts)
    
    # allocate bin-sized fisher matrix (same size as outer loop)
    fisher_per_bin = np.ones(bin_size) * np.nan

    # allocate 12 x 12 cov for use in inner loop
    invcov = np.zeros((F.bispec['pol_trpl'].size, F.bispec['pol_trpl'].size))

    # create (binned) inverse cov matrix for each ell
    # i.e. use the fact that 12x12 pol invcov can be factored
    # as (Cl-1)_l1^ip (Cl-1)_l2^jq (Cl-1)_l3^kr 
    invcov1 = np.ones((bin_size, 12, 12))
    invcov2 = np.ones((bin_size, 12, 12))
    invcov3 = np.ones((bin_size, 12, 12))

    f_check = 0

    for tidx_a, ptrp_a in enumerate(F.bispec['pol_trpl']):
        # ptrp_a = ijk
        for tidx_b, ptrp_b in enumerate(F.bispec['pol_trpl']):
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

    # Depending on lmin, start outer loop not at first bin.
    start_bidx = np.where(bins >= lmin)[0][0]

    end_bidx = np.where(bins >= min(lmax, bins[-1]))[0][0] + 1

    # loop same loop as in binned_bispectrum
    for idx1, i1 in enumerate(bins[start_bidx:end_bidx]):
        idx1 += start_bidx
        cl1 = invcov1[idx1,:,:] # 12x12

        # init
        fisher_per_bin[idx1] = 0.

        for idx2, i2 in enumerate(bins[idx1:end_bidx]):
            idx2 += idx1
            cl2 = invcov2[idx1,:,:] # 12x12

            cl12 = cl1 * cl2

            for idx3, i3 in enumerate(bins[idx2:end_bidx]):
                idx3 += idx2

                num = num_pass[idx1,idx2,idx3]
                if num == 0:
                    continue

                cl123 = cl12 * invcov3[idx3,:,:] #12x12

                B = bispec[idx1,idx2,idx3,:]

                f = np.einsum("i,ij,j", B, cl123, B)
#                f0 = np.einsum("i,i", B, B)
#                b0 = np.einsum("ij,ij", cl123, cl123)

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

#    print 'fisher_check:', f_check * (4*np.pi / np.sqrt(8))**2
#    print 'sigma:', 1/np.sqrt(f_check) * (np.sqrt(8)/4./np.pi)

    fisher_check = f_check * (4*np.pi / np.sqrt(8))**2
    sigma = 1/np.sqrt(f_check) * (np.sqrt(8)/4./np.pi)
    
    return fisher_check, sigma
    
#    for lidx, lmin in enumerate(range(2, 40)):
#        f = np.sum(fisher_per_bin[lmin-2:])
#        min_f.append(np.sqrt(f))



if __name__ == '__main__':

#    ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20181112_sst/' # S5
#    ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20181123_sst/'
    ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20181214_sst_debug/'
    out_dir = opj(ana_dir, 'fisher')
    camb_base = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst'
    camb_dir = opj(camb_base, 'camb_output/high_acy/sparse_5000')
    noise_base = '/mn/stornext/u3/adriaand/cmb_sst_ksw/ancillary/noise_curves'
#    noise_base = '/mn/stornext/u3/adriaand/cmb_sst_ksw/ancillary/noise_curves/so/v3/so'
#    lat_path = opj(noise_base, 's4/S4_2LAT_Tpol_default_noisecurves')
    lat_path = opj(noise_base, 'so/v3/so')
#    sac_path = noise_base
    sac_path = None

    # fixed
    lmin = 2
#    lmax = 4999
    lmax = 4000 # has to match beta etc
    lmax_f = 3000 # for fisher
    lmin_f = 250
#    A_lens = 0.13
    A_lens = 1.

    noise_amp_temp = 6.
    noise_amp_pol = 6 * np.sqrt(2)

    # NOTE
#    noise_amp_temp = .0
#    noise_amp_pol = .0 * np.sqrt(2)


    opts = {}
#    opts['nominal'] = dict(fsky=0.03, no_ee=False, no_tt=False, no_noise=False)
#    opts['no_ee'] = dict(fsky=0.03, no_ee=True, no_tt=False, no_noise=False)
#    opts['no_tt'] = dict(fsky=0.03, no_ee=False, no_tt=True, no_noise=False)
    opts['nominal'] = dict(fsky=1., no_ee=False, no_tt=False, no_noise=False)
    opts['no_ee'] = dict(fsky=1., no_ee=True, no_tt=False, no_noise=False)
    opts['no_tt'] = dict(fsky=1., no_ee=False, no_tt=True, no_noise=False)

    opts['cv_lim'] = dict(fsky=1., no_ee=False, no_tt=False, no_noise=True)
    opts['no_ee_cv_lim'] = dict(fsky=1., no_ee=True, no_tt=False, no_noise=True)
    opts['no_tt_cv_lim'] = dict(fsky=1., no_ee=False, no_tt=True, no_noise=True)

#    for template in ['local', 'equilateral']:            
    for template in ['local']:            
#        with open(opj(out_dir, 'fisher_{}.txt'.format(template)), 'w') as text_file:
        with open(opj(out_dir, 'fisher_so_{}.txt'.format(template)), 'w') as text_file:

            for key in opts:

                opt = opts[key]
                no_noise = opt.get('no_noise')
                fsky = opt.get('fsky')
                no_ee = opt.get('no_ee')
                no_tt = opt.get('no_tt')

                cls = get_cls(camb_dir, lmax, A_lens=A_lens)
                nls = get_nls(lat_path, lmax, sac_path=sac_path)                
                #nls = get_fiducial_nls(noise_amp_temp, noise_amp_pol, lmax)
                if no_noise:
                    nls *= 0.
                totcov = get_totcov(cls, nls, no_ee=no_ee, no_tt=no_tt)
                ells = np.arange(2, lmax+1)

#                plot_name = opj(out_dir, 'b_invcov_{}.png'.format(key))
                plot_name = opj(out_dir, 'b_so_invcov_{}.png'.format(key))

    #            for template in ['local', 'equilateral', 'orthogonal']:            

                text_file.write('template: {}\n'.format(template))
                text_file.write('option: {}\n'.format(key))
                text_file.write('no_noise: {}\n'.format(no_noise))
                text_file.write('fsky: {}\n'.format(fsky))
                text_file.write('A_lens: {}\n'.format(A_lens))
                text_file.write('no_ee: {}\n'.format(no_ee))
                text_file.write('no_tt: {}\n'.format(no_tt))

                fisher_check, sigma = run_fisher(template, 
                                                 ana_dir, camb_dir, totcov, ells,
                                                 lmin=lmin_f, lmax=lmax_f, fsky=fsky, 
                                                 plot_tag=plot_name, tag='r1_i10')

                text_file.write('fisher: {}\n'.format(fisher_check))
                text_file.write('sigma: {}\n'.format(sigma))
                text_file.write('\n')
