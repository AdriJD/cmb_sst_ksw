'''
Calculate elements of a Fisher matrix for a bispectrum analysis,
add optimal estimator later on
'''

import sys
import os
import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import trapz
from scipy.linalg import inv
from scipy.stats import binned_statistic
import camb_tools as ct
import pywigxjpf as wig
from beamconv import instrument as instr
import warnings

opj = os.path.join

class PreCalc(instr.MPIBase):
#class PreCalc(object):

    def __init__(self, camb_dir='.'):
        '''

        Keyword arguments
        -----------------
        camb_dir : str
            Path to directory containing camb output
            (Cls, transfers and aux) (default : ".")
        '''

        self.camb_dir = camb_dir
        self.depo = {}

        super(PreCalc, self).__init__()

    def get_camb_output(self, **kwargs):
        '''
        Store CAMB ouput in internal dictionaries.
        Loaded by root process and broadcasted to other ranks

        kwargs : {get_spectra_opts}
        '''
        source_dir = self.camb_dir

        lmax = None
        cls = None
        tr = None
        k = None

        # load spectra
        if self.mpi_rank == 0:

            cls, lmax = ct.get_spectra(source_dir, **kwargs)

        cls = self.broadcast_array(cls)

        lmax = self.broadcast(lmax)

        self.depo['cls'] = cls
        self.depo['cls_lmax'] = lmax

        # load transfer functions and aux
        for ttype in ['scalar', 'tensor']:

            if self.mpi_rank == 0:
                tr, lmax, k = ct.read_camb_output(source_dir, ttype=ttype)

            tr = self.broadcast_array(tr)
            k = self.broadcast_array(k)
            lmax = self.broadcast(lmax)

            self.depo[ttype] = {'transfer' : tr,
                                'lmax' : lmax,
                                'k' : k}

    def get_noise_curves(self, cross_noise=False, **kwargs):
        '''
        Load SO TT and pol noise curves and process
        into TT, EE, BB, TE, TB and EB noise curves

        Keyword Arguments
        ---------
        cross_noise : bool
            If set, TE, TB and EB elements get noise contribution
            (sqrt(nl_XX nl_YY) (default : False)
        tt_file : str
            Path to TT file
        pol_file : str
            Path to EE, BB file
        '''

        nls = None
        lmin = None
        lmax = None

        if self.mpi_rank == 0:

            if kwargs.get('sat_file'):
                
                ell_tt, nl_tt, ell_pol, nl_ee, nl_bb, nl_sat, ell_sat = ct.get_so_noise(
                    **kwargs)
            
            ell_tt, nl_tt, ell_pol, nl_ee, nl_bb = ct.get_so_noise(
                                                           **kwargs)

            # Make lmin and lmax equal
            lmin_tt = ell_tt[0]
            lmin_pol = ell_pol[0]
            lmax_tt = ell_tt[-1]
            lmax_pol = ell_pol[-1]

            lmin = max(lmin_tt, lmin_pol)
            lmax = min(lmax_tt, lmax_pol)

            # pol has lower lmin and higher lmax
            nl_ee = nl_ee[ell_pol >= lmin]
            nl_bb = nl_bb[ell_pol >= lmin]
            ell_pol = ell_pol[ell_pol >= lmin]

            nl_ee = nl_ee[ell_pol <= lmax]
            nl_bb = nl_bb[ell_pol <= lmax]
            ell_pol = ell_pol[ell_pol <= lmax]

            # Compute noise spectra: TT, EE, BB, TE, TB, EB
            nls = np.ones((6, ell_pol.size))
            nls[0] = nl_tt
            nls[1] = nl_ee
            nls[2] = nl_bb

            if cross_noise:
                nls[3] = np.sqrt(nl_tt * nl_ee)
                nls[4] = np.sqrt(nl_tt * nl_bb)
                nls[5] = np.sqrt(nl_ee * nl_bb)
            else:
                nls[3] = np.zeros_like(nl_bb)
                nls[4] = np.zeros_like(nl_bb)
                nls[5] = np.zeros_like(nl_bb)


        nls = self.broadcast_array(nls)
        lmin = self.broadcast(lmin)
        lmax = self.broadcast(lmax)

        self.depo['nls'] = nls
        self.depo['nls_lmin'] = int(lmin)
        self.depo['nls_lmax'] = int(lmax)


    def get_default_radii(self):
        '''
        Get the radii (in Mpc) used in table 1 of liquori 2007
        '''

        low = np.linspace(0, 9377, num=98, dtype=float, endpoint=False)
        re1 = np.linspace(9377, 10007, num=18, dtype=float, endpoint=False)
        re2 = np.linspace(10007, 12632, num=25, dtype=float, endpoint=False)
        rec = np.linspace(12632, 13682, num=300, dtype=float, endpoint=False)

        radii = np.concatenate((low, re1, re2, rec))

        self.depo['radii'] = radii
        return radii

    def get_updated_radii(self):
        '''
        Get the radii (in Mpc) that are more suitable to post-planck lcdm
        '''

        low = np.linspace(0, 9377, num=98, dtype=float, endpoint=False)
        re1 = np.linspace(9377, 10007, num=18, dtype=float, endpoint=False)
        re2 = np.linspace(10007, 12632, num=25, dtype=float, endpoint=False)
        rec = np.linspace(12632, 13682, num=50, dtype=float, endpoint=False)
        rec_new = np.linspace(13682, 14500, num=300, dtype=float, endpoint=False)
        rec_extra = np.linspace(14500, 18000, num=10, dtype=float, endpoint=False)

        radii = np.concatenate((low, re1, re2, rec, rec_new, rec_extra))

        self.depo['radii'] = radii
        return radii

    def init_beta(self, path, **kwargs):
        '''
        Load or compute beta (beta_s and beta_t)

        Arguments
        ---------
        path : str
            Path to beta files
        '''


    def beta(self, f=None, L_range=[-2, -1, 0, 1, 2], radii=None, bin=True,
             optimize=True):
        '''
        Calculate beta_l,L(r) = 2/pi * \int k^2 dk f(k) j_L(kr) T_X,l^(Z)(k).
        Vectorized. MPI'd by radius

        Arguments
        ---------

        Keyword arguments
        -----------------
        f : array-like
            Factor f(k) of (primordial) factorized shape function.
            Can be of shape (nfact, 3, k.size) If None, use local
            (default : None)
        radii : array-like
            Array with radii to compute. In units [Mpc], if None,
            use default_radii (default : None)
        L_range : array-like, int
            Possible deviations from ell, e.g. [-2, -1, 0, 1, 2]
            (default : [0])
        bin : bool
            Bin the resulting beta in ell. (default : True)
        Returns
        -------
        beta : array-like
            beta_ell_L (r) array of shape (r.size, lmax+1, L.size)
        '''

        if not self.depo['init_bins']:
            raise ValueError('bins not initialized')

        ells = self.ells
        L_range = np.asarray(L_range)

        if f is None:
            f = self.local()

        # you want to allow f to be of shape (nfact, 3, k.size)
        ndim = f.ndim
        if ndim == 3:
            f = f.copy()
        elif ndim == 1:
            f = f.copy()[np.newaxis, np.newaxis,:]
        elif ndim == 2:
            f = f.copy()[np.newaxis,:]
        else:
            raise ValueError('dimension {} of f not supported'.format(ndim))

        nfact = f.shape[0]
        ks = f.shape[1]
        print f.shape
        print ks

        if radii is None:
            radii = self.get_updated_radii()
        self.depo['radii'] = radii

        k = self.depo['scalar']['k']
        if k.size != f.shape[2]:
            raise ValueError('f and k not compatible: {}, {}'.format(
                    f.shape, k.shape))

        # scale f by k^2
        k2 = k**2
        f *= k2

        # allocate arrays for integral over k
        tmp_s = np.zeros_like(k)
        tmp_t = np.zeros_like(k)

        # load both scalar and tensor transfer functions
        transfer_s = self.depo['scalar']['transfer']
        transfer_t = self.depo['tensor']['transfer']
        pols_s = ['I', 'E']
        pols_t = ['I', 'E', 'B']

        # Distribute radii among cores
        # do a weird split for more even load balance, larger r is slower
        radii_per_rank = []

        radii_sub = radii[self.mpi_rank::self.mpi_size]
        for rank in xrange(self.mpi_size):
            radii_per_rank.append(radii[rank::self.mpi_size])

        # beta scalar and tensor
#        beta_s = np.zeros((ells.size, L_range.size, nfact, radii_sub.size, ks, len(pols_s)))
#        beta_t = np.zeros((ells.size, L_range.size, nfact, radii_sub.size, ks, len(pols_t)))

        beta_s = np.zeros((ells.size, L_range.size, nfact, ks, len(pols_s), radii_sub.size))
        beta_t = np.zeros((ells.size, L_range.size, nfact, ks, len(pols_t), radii_sub.size))

        # allocate space for bessel functions
        jL = np.zeros((L_range.size, k.size))

        # all arange with all possible L values
        ells_ext = np.arange(ells[-1] + L_range[-1] + 1)

        for ridx, radius in enumerate(radii_sub):

            kr = k * radius
            # Array that gives L -> kr idx mapping
            kr_idx = np.digitize(ells_ext, bins=kr, right=True)
            kr_idx[kr_idx == kr.size] = kr.size - 1

            print 'rank: {}, ridx: {}, radius: {} Mpc'.format(self.mpi_rank, ridx, radius)

            for lidx, ell in enumerate(ells):

                # loop over capital L's, only need 1, 3 or 5
                for Lidx, L in enumerate(L_range):
                    L = ell + L
                    if L < 0:
                        continue

                    # spherical bessel is zero for kr << L
                    # so dont waste time calculating those values
                    if optimize:
                        if L < 20:
                            kmin_idx = 0
                        elif L < 100:
                            kmin_idx = kr_idx[int(0.5 * L)]
                        elif L < 500:
                            kmin_idx = kr_idx[int(0.75 * L)]
                        elif L < 1000:
                            kmin_idx = kr_idx[int(0.8 * L)]
                        else:
                            kmin_idx = kr_idx[int(0.9 * L)]
                    else:
                        kmin_idx = 0

                    if lidx == 0:
                        # first pass, fill all
                        jL[Lidx,kmin_idx:] = spherical_jn(L, kr[kmin_idx:])
                    else:
                        # second pass only fill new row
                        if Lidx == L_range.size - 1:
                            jL[Lidx,:] *= 0.
                            jL[Lidx,kmin_idx:] = spherical_jn(L, kr[kmin_idx:])

                    # loop over T, E, B
                    for pidx, pol in enumerate(pols_t):

                        if pol != 'B':
                            tmp_s[kmin_idx:] = transfer_s[pidx,ell-2,kmin_idx:]
                            tmp_s[kmin_idx:] *= jL[Lidx,kmin_idx:]

                        tmp_t[kmin_idx:] = transfer_t[pidx,ell-2,kmin_idx:]
                        tmp_t[kmin_idx:] *= jL[Lidx,kmin_idx:]

                        for nidx in xrange(nfact):
                            for kidx in xrange(ks):

                                if pol != 'B':
                                    # scalars

                                    integrand_s = tmp_s[kmin_idx:] * f[nidx,kidx,kmin_idx:]
                                    b_int_s = trapz(integrand_s, k[kmin_idx:])
                                    beta_s[lidx,Lidx,nidx,kidx,pidx,ridx] = b_int_s

                                # tensors
                                integrand_t = tmp_t[kmin_idx:] * f[nidx,kidx,kmin_idx:]
                                b_int_t = trapz(integrand_t, k[kmin_idx:])
                                beta_t[lidx,Lidx,nidx,kidx,pidx,ridx] = b_int_t

                # permute rows such that oldest row can be replaced next ell
                jL = np.roll(jL, -1, axis=0)

        beta_s *= (2 / np.pi)
        beta_t *= (2 / np.pi)

        # Combine all sub range betas on root if mpi
        if self.mpi:
            self.barrier()

            # create full size beta on root
            if self.mpi_rank == 0:

                beta_s_full = np.zeros((ells.size, L_range.size, nfact,
                                        ks, len(pols_s), radii.size))
                beta_t_full = np.zeros((ells.size, L_range.size, nfact,
                                        ks, len(pols_t), radii.size))

                # already place root beta sub into beta_full
                for ridx, radius in enumerate(radii_per_rank[0]):
                    # find radius index in total radii
                    ridx_tot, = np.where(radii == radius)[0]
                    beta_s_full[:,:,:,:,:,ridx_tot] = beta_s[:,:,:,:,:,ridx]
                    beta_t_full[:,:,:,:,:,ridx_tot] = beta_t[:,:,:,:,:,ridx]

            else:
                beta_s_full = None
                beta_t_full = None

            # loop over all non-root ranks
            for rank in xrange(1, self.mpi_size):

                # allocate space for sub beta on root
                if self.mpi_rank == 0:
                    r_size = radii_per_rank[rank].size

                    beta_s_sub = np.ones((ells.size, L_range.size, nfact, ks, len(pols_s), r_size))
                    beta_t_sub = np.ones((ells.size, L_range.size, nfact, ks, len(pols_t), r_size))

                    beta_s_sub *= np.nan
                    beta_t_sub *= np.nan

                # send beta_sub to root
                if self.mpi_rank == rank:
                    self._comm.Send(beta_s, dest=0, tag=rank)
                    self._comm.Send(beta_t, dest=0, tag=rank + self.mpi_size + 1)

                if self.mpi_rank == 0:
                    self._comm.Recv(beta_s_sub, source=rank, tag=rank)
                    self._comm.Recv(beta_t_sub, source=rank, tag=rank + self.mpi_size + 1)

                    # place into beta_full
                    for ridx, radius in enumerate(radii_per_rank[rank]):

                        # find radius index in total radii
                        ridx_tot, = np.where(radii == radius)[0]

                        beta_s_full[:,:,:,:,:,ridx_tot] = beta_s_sub[:,:,:,:,:,ridx]
                        beta_t_full[:,:,:,:,:,ridx_tot] = beta_t_sub[:,:,:,:,:,ridx]


            # broadcast full beta array to all ranks
            beta_s = self.broadcast_array(beta_s_full)
            beta_t = self.broadcast_array(beta_t_full)

        if not bin:
            self.depo['scalar']['beta'] = beta_s
            self.depo['tensor']['beta'] = beta_t
            return

        # Bin beta
        bins = self.bins
        indices = np.digitize(ells, bins, right=False) - 1

        beta_s_f = np.asfortranarray(beta_s)
        beta_t_f = np.asfortranarray(beta_t)

        b_beta_s_f = np.zeros((bins.size,L_range.size,nfact,ks,len(pols_s),radii.size))
        b_beta_t_f = np.zeros((bins.size,L_range.size,nfact,ks,len(pols_t),radii.size))

        b_beta_s_f = np.asfortranarray(b_beta_s_f)
        b_beta_t_f = np.asfortranarray(b_beta_t_f)

        for pidx, pol in enumerate(pols_t):
            for kidx in xrange(ks):
                for ridx, radius in enumerate(radii):
                    for nidx in xrange(nfact):
                        for Lidx, L in enumerate(L_range):

                            if pol != 'B':
                                # scalar
                                tmp_beta = beta_s_f[:,Lidx,nidx,kidx,pidx,ridx]

                                b_beta_s_f[:-1,Lidx,nidx,kidx,pidx,ridx], _, _ = \
                                    binned_statistic(ells, tmp_beta, statistic='mean',
                                                     bins=bins)

                                # expand to full size
                                beta_s_f[:,Lidx,nidx,kidx,pidx,ridx] = \
                                    b_beta_s_f[indices,Lidx,nidx,kidx,pidx,ridx]


                            # tensor
                            tmp_beta = beta_t_f[:,Lidx,nidx,kidx,pidx,ridx]

                            b_beta_t_f[:-1,Lidx,nidx,kidx,pidx,ridx], _, _ = \
                                binned_statistic(ells, tmp_beta, statistic='mean',
                                                 bins=bins)

                            # expand to full size
                            beta_t_f[:,Lidx,nidx,kidx,pidx,ridx] = \
                                b_beta_t_f[indices,Lidx,nidx,kidx,pidx,ridx]


        beta_s = np.ascontiguousarray(beta_s_f)
        beta_t = np.ascontiguousarray(beta_t_f)

        b_beta_s = np.ascontiguousarray(b_beta_s_f)
        b_beta_t = np.ascontiguousarray(b_beta_t_f)

        self.depo['scalar']['beta'] = beta_s
        self.depo['tensor']['beta'] = beta_t

        # binned versions
        self.depo['scalar']['b_beta'] = b_beta_s
        self.depo['tensor']['b_beta'] = b_beta_t

        return

#class Experiment(object):
#    '''
#    Experimental details such as noise, fsky, pol
#    '''
#
#    def __init__(self, fsky=1.):
#
#        self.fsky = fsky
#
#    def get_noise(self):
#        '''
#        Load noise curves
#        '''

        # column colin: [ell] [N_ell^TT in uK^2] [N_ell^yy (dimensionless)]

class Bispectrum(PreCalc):
    '''
    Create arrays of shape (nfact, 3, k.size)
    '''

    def __init__(self, **kwargs):

        self.scalar_amp = 2.1e-9
        self.ns = 0.96
        self.nt = 0
        self.r = 0.03

        super(Bispectrum, self).__init__(**kwargs)

        self.depo['init_bins'] = False

    def local(self, fnl=1):
        '''
        eq. 18 from Meerburg 2016 without the angular
        dependent parts and I = local
        '''

        amp = (2 * np.pi)**3 * 16 * np.pi**4 * self.scalar_amp * np.sqrt(self.r)
        amp *= fnl

        ks = self.depo['scalar']['k']
        kt = self.depo['tensor']['k']

        km3 = ks**-3
        ones = np.ones(ks.size)

#        template = np.asarray([[km3, km3, ones], [km3, ones, km3], [ones, km3,km3]])
        km3 *= amp
        template = np.asarray([km3, ones])
#        template *= amp

        return template

    def equilateral(self, fnl=1):
        '''

        '''

        pass

    def orthogonal(self, fnl=1):
        '''

        '''
        pass

#class Fisher(Bispectrum, Experiment):
class Fisher(Bispectrum):

    def __init__(self, **kwargs):
        tag = kwargs.pop('tag')
        lensed = kwargs.pop('lensed')
        camb_opts = dict(tag=tag, lensed=lensed)

        super(Fisher, self).__init__(**kwargs)

        self.get_camb_output(**camb_opts)

        # Initialize wigner tables for lmax=8000
        wig.wig_table_init(2 * 8000, 9)
        wig.wig_temp_init(2 * 8000)


    def init_bins(self, bins=None, parity='odd'):
        '''
        Create default ell bins. Stores attributes for
        unique_ells, bins, num_pass and first_pass

        Keyword Arguments
        -----------------
        bins : array-like
            Lower value of bins in ell
        parity : str
            Consider tuples in ell that are parity "odd"
            or "even". If None, use both. (default : "odd")

        Notes
        -----
            Sum over ell is l1 <= l2 <= l3

            unique_ells : array-like
                Values of ell that are used (for alpha, beta)
            bins : array-like
                Bin multipoles (same as Meerbug 2016)
            num_pass : array-like
                   number of tuples per 3d bin that pass triangle
                   condition. No parity check right now.
                   SIGMAi1i2i3 in bucher 2015.
                   Shape : (nbins, nbins, nbins)
            first_pass : array-like
                   Lowest sum tuple per 3d bin that passes triangle
                   condition. Shape : (nbins, nbins, nbins, 3)
        '''

        # Determine lmin and lmax
        lmax_s = self.depo['scalar']['lmax']
        lmax_t = self.depo['tensor']['lmax']

        lmax_cl = self.depo['cls_lmax']

        lmax_nl = self.depo['nls_lmax']
        lmin_nl = self.depo['nls_lmin']

        # store parity
        self.depo['parity'] = parity

        lmax = min(lmax_s, lmax_t, lmax_cl, lmax_nl)
        lmin = lmin_nl

        self.lmax = lmax
        self.lmin = lmin

        lmax = 500 ########## NOTENOTE

        if lmax < lmin:
            raise ValueError('lmax < lmin')

        # bins used in Meerburg 2016
        bins_0 = np.arange(lmin, 101, 1)
        bins_1 = np.arange(110, 510, 10)
        bins_2 = np.arange(520, 8000, 20)
        bins = np.concatenate((bins_0, bins_1, bins_2))

        max_bin = np.argmax(bins>lmax)
        bins = bins[:max_bin]

        num_bins = bins.size
        # allocate space for number of good tuples per 3d bin
        num_pass = np.zeros((num_bins, num_bins, num_bins), dtype=int)
        # allocate space for first good tuple per 3d bin
        first_pass = np.zeros((num_bins, num_bins, num_bins, 3), dtype=int)

        if parity == 'odd':
            pmod = 1
        elif parity == 'even':
            pmod = 0
        else:
            pmod = None

        # calculate bins in parallel, split ell in outer loop
        ells = np.arange(lmin, lmax+1)
        self.ells = ells
        ells_sub = ells[self.mpi_rank::self.mpi_size]

        # create an ell -> bin idx lookup table for 2nd loop
        idx = np.digitize(ells, bins, right=False) - 1

        # boolean index array for inner loop
        ba = np.ones(ells.size, dtype=bool)

        for ell1 in ells_sub:
#            print 'rank: {}, ell: {}'.format(self.mpi_rank, ell1)
            # which bin?
            idx1 = np.argmax(ell1 < bins) - 1

            for ell2 in xrange(ell1, lmax+1):

                # which bin?
                idx2 = idx[ell2 - lmin]

                # exclude ells below ell2
                ba[:ell2-lmin+1] = False

                # exclude parity odd/even
                if parity:
                    if (ell1 + ell2) % 2:
                        # sum is odd, l3 must be even if parity=odd
                        ba[ells%2 == pmod] *= False
                    else:
                        # sum is even, l3 must be odd if parity=odd
                        ba[~(ells%2 == pmod)] *= False

                # exclude triangle
                ba[(abs(ell1 - ell2) > ells) | (ells > (ell1 + ell2))] *= False

                # use boolean index array to determine good ell3s
                gd_ells = ells[ba]

                # find number of good ells within bin
                bin_idx = idx[ba] # for each good ell, index of corr. bin
                # find position in bin_idx of first unique index
                u_idx, first_idx, count = np.unique(bin_idx, return_index=True,
                                          return_counts=True)

                num_pass[idx1,idx2,u_idx] = count
                # first pass needs a (u_idx.size, 3) array
                first_pass[idx1,idx2,u_idx,0] = ell1
                first_pass[idx1,idx2,u_idx,1] = ell2
                first_pass[idx1,idx2,u_idx,2] = gd_ells[first_idx]

                # reset array
                ba[:] = True

        # now combine num_pass and first_pass on root
        if self.mpi:
            self.barrier()

            # allocate space to receive arrays on root
            if self.mpi_rank == 0:
                num_pass_rec = np.zeros_like(num_pass)
                first_pass_rec = np.zeros_like(first_pass)

            # loop over all non-root ranks
            for rank in xrange(1, self.mpi_size):

                # send arrays to root
                if self.mpi_rank == rank:
                    self._comm.Send(num_pass, dest=0, tag=rank)
                    self._comm.Send(first_pass, dest=0, tag=rank + self.mpi_size)

                if self.mpi_rank == 0:
                    self._comm.Recv(num_pass_rec, source=rank, tag=rank)
                    self._comm.Recv(first_pass_rec, source=rank, tag=rank + self.mpi_size)

                    # simply add num_pass but only take first pass if lower
                    num_pass += num_pass_rec

                    sum_root = np.sum(first_pass, axis=3)
                    sum_rec = np.sum(first_pass_rec, axis=3)

                    mask = sum_root == 0
                    first_pass[mask,:] = first_pass_rec[mask,:]

                    mask2 = sum_rec < sum_root
                    # but rec is not zero
                    mask2 *= sum_rec != 0
                    first_pass[mask2,:] = first_pass_rec[mask2,:]

            # broadcast full arrays to all ranks
            num_pass = self.broadcast_array(num_pass)
            first_pass = self.broadcast_array(first_pass)

        self.barrier()

        # trim away the zeros in first_pass
        self.unique_ells = np.unique(first_pass)[1:]
        self.bins = bins
        self.num_pass = num_pass
        self.first_pass = first_pass

        self.depo['init_bins'] = True

    def get_binned_invcov(self):
        '''
        Combine singnal and noise into an inverse cov matrix
        per bin. We first bin, then compute inverse.

        Notes
        -----
        Stores shape = (ells, 3, 3) array. ells is unbinned.
        '''

        ells = self.ells
        indices = np.digitize(ells, self.bins, right=False) - 1

        cls = self.depo['cls'] # Signal, lmin = 2
        nls = self.depo['nls'] # Noise

        lmin = ells[0]
        lmax = ells[-1]

        # assume nls start at lmin, cls at ell=2
        nls = nls[:,:(lmax - lmin + 1)]
        cls = cls[:,lmin-2:lmax-1]

        # Add signal cov to noise (cls for TB and TE not present in cls)
        nls[:4,:] += cls

        bins = self.bins

        bin_cov = np.ones((bins.size, 3, 3))
        bin_cov *= np.nan
        bin_invcov = bin_cov.copy()

        nls_dict = {'TT': 0, 'EE': 1, 'BB': 2, 'TE': 3,
                    'ET': 3, 'BT': 4, 'TB': 4, 'EB': 5,
                    'BE': 5}

        for pidx1, pol1 in enumerate(['T', 'E', 'B']):
            for pidx2, pol2 in enumerate(['T', 'E', 'B']):

                # Cl+Nl array
                nidx = nls_dict[pol1+pol2]
                nell = nls[nidx,:]

                # Bin
                bin_cov[:-1,pidx1,pidx2], _, _ = binned_statistic(ells, nell,
                                                             statistic='mean',
                                                             bins=bins)

        # Invert
        for bidx in xrange(bins.size - 1):
            bin_invcov[bidx,:] = inv(bin_cov[bidx,:])

        # Expand binned inverse cov and cov to full size again
        invcov = np.ones((ells.size, 3, 3))
        invcov *= np.nan
        cov = invcov.copy()
        invcov[:] = bin_invcov[indices,:]
        cov[:] = bin_cov[indices,:]

        self.invcov = invcov
        self.cov = cov
        self.bin_cov = bin_cov
        self.ells = ells
        self.nls = nls

    def init_wig3j(self):
        '''
        Precompute I^000_lL1 and I^20-2_lL2 for all unique ells
        and \Delta L \in [-1, 1] and [-2, -1, 0, 1, 2]

        Store internally as wig_s and wig_t
        shape = (ells.size, \Delta L)
        '''

        u_ells = self.unique_ells
        ells = self.ells

#        u_ells = np.arange(40, 200)
#        ells = np.arange(40, 200)

        lmin = ells[0]

        wig_s = np.zeros((ells.size, 5))
        wig_t = np.zeros((ells.size, 5))

        for ell in u_ells:
            lidx = ell - lmin

#            for Lidx, DL in enumerate([-1, 1]):
            for Lidx, DL in enumerate([-2, -1, 0, 1, 2]):

                L = ell + DL
                if DL == -1 or DL == 1:

                    tmp = wig.wig3jj([2*ell, 2*L, 2,
                                      0, 0, 0])
                    tmp *= np.sqrt((2 * L + 1) * (2 * ell + 1) * 3)
                    tmp /= (2 * np.sqrt(np.pi))
                    wig_s[lidx,Lidx] = tmp

#            for Lidx, DL in enumerate([-2, -1, 0, 1, 2]):
#                L = ell + DL

                tmp = wig.wig3jj([2*ell, 2*L, 4,
                                  4, 0, -4])
                tmp *= np.sqrt((2 * L + 1) * (2 * ell + 1) * 5)
                tmp /= (2 * np.sqrt(np.pi))

                wig_t[lidx,Lidx] = tmp

        self.wig_s = wig_s
        self.wig_t = wig_t

    def init_pol_triplets(self):
        '''
        Store polarization triples internally

        I = 0, E = 1, B = 2
        '''

        pol_trpl = np.zeros((12, 3), dtype=int)

        pol_trpl[0] = 0, 0, 2
        pol_trpl[1] = 0, 2, 0
        pol_trpl[2] = 2, 0, 0
        pol_trpl[3] = 1, 0, 2
        pol_trpl[4] = 1, 2, 0
        pol_trpl[5] = 2, 1, 0
        pol_trpl[6] = 0, 1, 2
        pol_trpl[7] = 0, 2, 1
        pol_trpl[8] = 2, 0, 1
        pol_trpl[9] = 1, 1, 2
        pol_trpl[10] = 1, 2, 1
        pol_trpl[11] = 2, 1, 1

        self.pol_trpl = pol_trpl


    def binned_bispectrum(self, DL1, DL2, DL3):
        '''
        Compute B for each bin

        Save as (bins, pol1, pol2, pol3, B)

        DL1 : Delta L
        '''

        # loop over bins
        bins = self.bins
        ells = self.ells
        lmin = ells[0]
        u_ells = self.unique_ells
        num_pass = self.num_pass
        # convert to float
        num_pass = num_pass.astype(float)
        first_pass = self.first_pass

        wig_t = self.wig_t
        wig_s = self.wig_s

        pol_trpl = self.pol_trpl
        psize = pol_trpl.shape[0]

        # binned betas
        beta_s = self.depo['scalar']['b_beta']
        beta_t = self.depo['tensor']['b_beta']

        # get radii corresponing to beta
        radii = self.depo['radii']
        r2 = radii**2

        # allocate r arrays
        integrand = np.zeros_like(radii)
        integrand_tss = integrand.copy()
        integrand_sts = integrand.copy()
        integrand_sst = integrand.copy()

        # allocate bispectrum
        nbins = bins.size
        bispectrum = np.zeros((nbins, nbins, nbins, psize))

        # check parity
        parity = self.depo['parity'] 
        if parity == 'odd' and (DL1 + DL2 + DL3) % 2 == 0:
            warnings.warn('parity is odd and DL1 + DL2 + DL3 is even, '
                          'bispectrum is zero')
            return

        elif parity == 'even' and (DL1 + DL2 + DL3) % 2 == 1:
            warnings.warn('parity is even and DL1 + DL2 + DL3 is odd, '
                          'bispectrum is zero')
            return

        import time

        Lidx1 = DL1 + 2
        Lidx2 = DL2 + 2
        Lidx3 = DL3 + 2

        # define function names locally for faster lookup
        wig3jj = wig.wig3jj
        wig9jj = wig.wig9jj
        trapz_loc = trapz


        # Scatter outer bins over ranks
        if self.mpi:

            bins_outer = bins.copy()
            # remove last bin
            bins_outer = bins_outer[:-1]
            idx_outer = np.arange(bins_outer.size)

            # scatter
            bins_outer = bins_outer[self.mpi_rank::self.mpi_size]
            idx_outer = idx_outer[self.mpi_rank::self.mpi_size]
            
        else:
            # remove last bin
            bins_outer = bins.copy()[:-1]
            idx_outer = np.arange(bins_outer.size)

        t0 = time.time()
        # Note, we do not consider the last bin
#        for idx1, i1 in enumerate(bins[:-1]):
        for idx1, i1 in zip(idx_outer, bins_outer):
            print self.mpi_rank, idx1
            # load binned beta
            beta_s_l1 = beta_s[idx1,Lidx1,0,0,:,:] # (2,r.size)
            beta_t_l1 = beta_t[idx1,Lidx1,0,0,:,:] # (3,r.size)

            alpha_s_l1 = beta_s[idx1,Lidx1,0,1,:,:] # (2,r.size)
            alpha_t_l1 = beta_t[idx1,Lidx1,0,1,:,:] # (3,r.size)


            for idx2, i2 in enumerate(bins[idx1:-1]):
                idx2 += idx1

                # load binned beta
                beta_s_l2 = beta_s[idx2,Lidx2,0,0,:,:] # (2,r.size)
                beta_t_l2 = beta_t[idx2,Lidx2,0,0,:,:] # (3,r.size)

                alpha_s_l2 = beta_s[idx2,Lidx2,0,1,:,:] # (2,r.size)
                alpha_t_l2 = beta_t[idx2,Lidx2,0,1,:,:] # (3,r.size)

                for idx3, i3 in enumerate(bins[idx2:-1]):
                    idx3 += idx2

                    num = num_pass[idx1, idx2, idx3]
                    if num == 0.:
                        # not a valid tuple
                        continue

                    # load binned beta
                    beta_s_l3 = beta_s[idx3,Lidx3,0,0,:,:] # (2,r.size)
                    beta_t_l3 = beta_t[idx3,Lidx3,0,0,:,:] # (3,r.size)

                    alpha_s_l3 = beta_s[idx3,Lidx3,0,1,:,:] # (2,r.size)
                    alpha_t_l3 = beta_t[idx3,Lidx3,0,1,:,:] # (3,r.size)

                    ell1, ell2, ell3 = first_pass[idx1, idx2, idx3,:]

                    # indices to full-size ell arrays
                    lidx1 = ell1 - lmin
                    lidx2 = ell2 - lmin
                    lidx3 = ell3 - lmin

                    L1 = DL1 + ell1
                    L2 = DL2 + ell2
                    L3 = DL3 + ell3

                    # Overall angular part
                    ang = wig3jj([2*L1, 2*L2, 2*L3,
                                  0, 0, 0])
                    ang *= np.real((-1j)**(ell1 + ell2 + ell3 - 1)) 
                    ang *= (-1)**((L1 + L2 + L3)/2)
                
                    # calculate the angular parts for each S, S, T comb
                    # TSS
                    ang_tss = wig_t[lidx1, Lidx1]
#                    print ang_tss
                    ang_tss *= wig_s[lidx2, Lidx2]
#                    print ang_tss
                    ang_tss *= wig_s[lidx3, Lidx2]
#                    print ang_tss
                    ang_tss *= ang
                    if ang_tss != 0.:
#                        print 'tss'
                        ang_tss *= wig9jj( [(2*ell1),  (2*ell2),  (2*ell3),
                                                (2*L1),  (2*L2),  (2*L3),
                                                4,  2,  2] )
                    else:
                        pass

                    # STS
                    ang_sts = wig_s[lidx1, Lidx2]
                    ang_sts *= wig_t[lidx2, Lidx2]
                    ang_sts *= wig_s[lidx3, Lidx2]
#                    print ang_tss
                    ang_sts *= ang
                    if ang_sts != 0.:
#                        print 'sts'
                        ang_sts *= wig9jj( [(2*ell1),  (2*ell2),  (2*ell3),
                                                (2*L1),  (2*L2),  (2*L3),
                                                2, 4,  2] )
                    else:
                        pass
                    # TSS
                    ang_sst = wig_s[lidx1, Lidx3]
                    ang_sst *= wig_s[lidx2, Lidx3]
                    ang_sst *= wig_t[lidx3, Lidx3]
#                    print ang_tss
                    ang_sst *= ang
                    if ang_sst != 0.:
 #                       print 'sst'
                        ang_sst *= wig9jj( [(2*ell1),  (2*ell2),  (2*ell3),
                                                (2*L1),  (2*L2),  (2*L3),
                                                2,  2,  4] )
                    else:
                        pass

                    if ang_tss == 0. and ang_sts == 0. and ang_sst == 0.:
                        # wrong L,ell comb
                        print ell1, ell2, ell3, L1, L2, L3
                        print ang_tss, ang_sts, ang_sst
                        print np.abs(ell1 - ell2) <= ell3 <= ell1 + ell2
                        print np.abs(L1 - L2) <= L3 <= L1 + L2
                        print np.abs(ell1 - L1) <= 2 <= ell1 + L1
                        print np.abs(ell1 - L1) <= 4 <= ell1 + L1
                        print np.abs(ell2 - L2) <= 2 <= ell2 + L2
                        print np.abs(ell2 - L2) <= 4 <= ell2 + L2
                        print np.abs(ell3 - L3) <= 2 <= ell3 + L3
                        print np.abs(ell3 - L3) <= 4 <= ell3 + L3

                        continue

                    # loop over pol combs
                    for pidx in xrange(psize):

                        pidx1, pidx2, pidx3 = pol_trpl[pidx,:]

                        # integrate over bba + bab + abb for each T, S, S comb
                        integrand[:] = 0.

                        # TSS
                        if pidx2 == 2 or pidx3 == 2 or ang_tss == 0.:
                            # no B-mode for scalar or this l, L combination
                            pass
                        else:
                            integrand_tss[:] = beta_t_l1[pidx1,:] * beta_s_l2[pidx2,:] * alpha_s_l3[pidx3,:]
                            integrand_tss += beta_t_l1[pidx1,:] * alpha_s_l2[pidx2,:] * beta_s_l3[pidx3,:]
                            integrand_tss += alpha_t_l1[pidx1,:] * beta_s_l2[pidx2,:] * beta_s_l3[pidx3,:]
                            integrand_tss *= ang_tss
                            
                            integrand += integrand_tss

                        # STS
                        if pidx1 == 2 or pidx3 == 2 or ang_sts == 0.:
                            # no B-mode for scalar
                            pass
                        else:
                            integrand_sts[:] = beta_s_l1[pidx1,:] * beta_t_l2[pidx2,:] * alpha_s_l3[pidx3,:]
                            integrand_sts += beta_s_l1[pidx1,:] * alpha_t_l2[pidx2,:] * beta_s_l3[pidx3,:]
                            integrand_sts += alpha_s_l1[pidx1,:] * beta_t_l2[pidx2,:] * beta_s_l3[pidx3,:]
                            integrand_sts *= ang_sts

                            integrand += integrand_sts

                        # SST
                        if pidx1 == 2 or pidx2 == 2 or ang_sst == 0.:
                            # no B-mode for scalar
                            pass
                        else:
                            integrand_sst[:] = beta_s_l1[pidx1,:] * beta_s_l2[pidx2,:] * alpha_t_l3[pidx3,:]
                            integrand_sst += beta_s_l1[pidx1,:] * alpha_s_l2[pidx2,:] * beta_t_l3[pidx3,:]
                            integrand_sst += alpha_s_l1[pidx1,:] * beta_s_l2[pidx2,:] * beta_t_l3[pidx3,:]
                            integrand_sst *= ang_sst

                            integrand += integrand_sst
                        
                        # Integrate over r
                        integrand *= r2
                        bispec = trapz_loc(integrand, radii)

                        # divide by num (note, already floats)
                        bispec /= num

                        # divide by Delta_i1i2i3
                        if i1 == i2 == i3:
                            bispec /= 6.
                        elif i1 != i2 != i3:
                            pass
                        else:
                            bispec /= 2.
                        
                        bispectrum[idx1,idx2,idx3,pidx] = bispec                                    
            
        bispectrum *= (8 * np.pi)**(3/2.) / 3.

        if not self.mpi:            
            return bispectrum

        else:
            # sum all bispectra on root
            self.barrier()

            # create empty beta for receiving on root
            if self.mpi_rank == 0:

                bispec_rec = np.zeros_like(bispectrum)

            # loop over all non-root ranks
            for rank in xrange(1, self.mpi_size):

                # send bispectrum to root
                if self.mpi_rank == rank:
                    self._comm.Send(bispectrum, dest=0, tag=rank)

                if self.mpi_rank == 0:
                    self._comm.Recv(bispec_rec, source=rank, tag=rank)

                    # place into root bispectrum
                    bispectrum += bispec_rec
                    bispec_rec[...] = 0.

            return bispectrum


#                    print 'b', tb - time.time()
#                    print 'c', tc - time.time()
#                        print '0', rad_int, '1', ang_tss, '2', ang_sts, '3', ang_sst

                        # factor (-i)^sum(ell) in eq. 6.19 shiraishi. Cancels with earlier factor
#                        rad_int *=

#                        bispectrum[idx1,idx2,idx3,pidx] =



        print time.time() - t0

        # load up

    def get_Ls(self, ell1, ell2, ell3, prim_type):
        '''
        Arguments
        ---------
        ell1, ell2, ell3 : int
            Multipoles
        prim_type : str
            Primoridal 3-point type. Either 'tss', 'sts', 'sst'

        Returns
        -------
        Ls : array-like
            Shape (20, 3), nan where triangle cond is not met
            or sum is odd.
        '''
        Ls = np.empty((20, 3))

        if prim_type == 'tss':

            Ls1 = range(ell1 - 2, ell1 + 3)
            Ls2 = [ell2 - 1, ell2 + 1]
            Ls3 = [ell3 - 1, ell3 + 1]

        elif prim_type == 'sts':

            Ls1 = [ell1 - 1, ell1 + 1]
            Ls2 = range(ell2 - 2, ell2 + 3)
            Ls3 = [ell3 - 1, ell3 + 1]

        elif prim_type == 'sst':

            Ls1 = [ell1 - 1, ell1 + 1]
            Ls2 = [ell2 - 1, ell2 + 1]
            Ls3 = range(ell3 - 2, ell3 + 3)

        n = 0
        for i, L3 in enumerate(Ls1):
            for j, L2 in enumerate(Ls2):
                for k, L1 in enumerate(Ls3):

                    if (L1 + L2 + L3) % 2:
                        # sum is odd
                        Ls[n,:] = np.nan
                        n += 1
                        continue

                    # triangle condition on Ls
                    elif not abs(L1 - L2) <= L3:
                        Ls[n,:] = np.nan
                        n += 1
                        continue

                    elif not L3 <= (L1 + L2):
                        Ls[n,:] = np.nan
                        n += 1
                        continue

                    Ls[n] = L1, L2, L3

        # how does the fisher loop know which L's to skip
        # depending on pol? I guess another if statemant there
        return Ls


    def binned_fisher(self):
        '''
        Loop over bins, and compute fisher
        '''



        # probably MPI ell bins


        pass




    def init_ell(self):
        '''
        Precompute all valid l1, l2, l3 and 1/Delta
        l1, l2, l3 already pass triangle condition
        '''

        lmax_s = self.depo['scalar']['lmax']
        lmax_t = self.depo['tensor']['lmax']

        ells = []
        idelta = []

        print 'determining l1, l2, l3 and 1/Delta'

        for l1 in xrange(2, lmax_s+1):

            for l2 in xrange(2, l1+1):
                for l3 in xrange(2, min(l2, lmax_t)+1):
                    if np.abs(l1 - l2) <= l3 <= l1 + l2:
                        if l1 != l2 != l3:
                            idelta.append(1)
                        elif l1 == l2 == l3:
                            idelta.append(1/6.)
                        else:
                            idelta.append(0.5)

                        ells.append([l1, l2, l3])


        ells = np.asarray(ells)
        ells = np.ascontiguousarray(ells)

        idelta = np.asarray(idelta)
        idelta = np.ascontiguousarray(idelta)

        print 'done'

        self.ells = ells
        self.idelta = idelta

        return ells, idelta

    def init_Ls(self):
        '''
        Get L1, L2, L3 that pass triangle condition
        with themselves and other 3 wigner 3js.
        '''

        print 'determining L1, L2, L3'
        num_ells = self.ells.shape[0]
        print num_ells
        l1 = self.ells[:,0]
        l2 = self.ells[:,1]
        l3 = self.ells[:,2]

        Ls = np.zeros((num_ells, 20, 3), dtype=int) # 2 * 2 * 5
        tri_cond = np.ones((num_ells, 20), dtype=bool)

        for i, dL3 in enumerate([-2, -1, 0, 1, 2]):
            for j, dL2 in enumerate([-1, 1]):
                for k, dL1 in enumerate([-1, 1]):

                    L1 = l1 + dL1
                    L2 = l2 + dL2
                    L3 = l3 + dL3

                    idx = 9 * i + 3 * j + k

                    Ls[:,idx,0] = L1
                    Ls[:,idx,1] = L2
                    Ls[:,idx,2] = L3

                    # internal triangle cond
                    tri_cond[:,idx] *= np.abs(L1 - L2) <= L3
                    tri_cond[:,idx] *= L3 <= (L1 + L2)

                    # sum must be even
                    even_cond = np.mod(L1 + L2 + L3, 2) - 1
                    tri_cond[:,idx] *= even_cond.astype(bool)

        Ls = np.ascontiguousarray(Ls)
        tri_cond = np.ascontiguousarray(tri_cond)

        print 'done'

        self.Ls = Ls
        self.tri_cond = tri_cond
        return Ls, tri_cond



    def inverse_delta(self, l1, l2, l3):

        if l1 != l2 != l3:
            return 1.
        elif l1 == l2 == l3:
            return 1/6.
        else:
            return 0.5

#    def get_Ls(self, l1, l2, l3, DL1, DL2, DL3):
#        '''
#        DL : array-like
#            e.g. [-2, -1, 0, 1, 2]
#        '''

#        L1 = DL1 + l1
#        L2 = DL2 + l2
#        L3 = DL3 + l3

#        return np.array(np.meshgrid(L1, L2, L3)).T.reshape(-1,3)

    def fisher_new(self):

        lmax_s = self.depo['scalar']['lmax']
        lmax_t = self.depo['tensor']['lmax']

        fisher_tot = 512 * np.pi**3 / 9.

        DL1 = np.array([-1, 1])
        DL2 = np.array([-1, 1])
        DL3 = np.array([-2, -1, 0, 1, 2])

        # to store H_L1L2L3 for given l1 l2 l3
        tmp = np.zeros((DL1.size * DL2.size * DL3.size))

        # arguments for 9j
        arg9j = np.zeros(9, dtype=int)
        arg9j[6:10] = 2, 2, 4

        Lidx = np.array(np.meshgrid(np.arange(DL1.size),
                                    np.arange(DL2.size),
                                    np.arange(DL3.size))).T.reshape(-1,3)

        for l1 in xrange(2, lmax_s+1):
            idx1 = l1 - 2
            print l1
            for l2 in xrange(2, l1+1):
                idx2 = l2 - 2

                for l3 in xrange(2, min(l2, lmax_t)+1):

                    if not np.abs(l1 - l2) <= l3 <= l1 + l2:
                        continue

                    idx3 = l3 - 2

                    fisher = self.inverse_delta(l1, l2, l3)

                    arg9j[0:3] = l1, l2, l3
                    arg9j[0:3] *= 2

                    # get L1, L2, L3, L4, L5, L6
                    Ls = self.get_Ls(l1, l2, l3, DL1, DL2, DL3)

                    for idx, (L1, L2, L3) in enumerate(Ls):

                        # L1 L2 L3 0 0 0 wigner
                        tmp[idx] = wig.wig3jj([2*L1, 2*L2, 2*L3, 0, 0, 0])
#                        tmp = 1
                        # triangle condition on L1 L2 L3 and sum(L) is even
                        if not tmp[idx]:
                            continue

                        # 9j
                        arg9j[3:6] = L1, L2, L3
                        arg9j[3:6] *= 2

                        tmp[idx] *= wig.wig9jj(arg9j)

                        # triangle condition on 9j
                        if not tmp[idx]:
                            continue

                        # find the indices of pre_s and pre_t
                        L1idx = Lidx[idx][0]
                        L2idx = Lidx[idx][1]
                        L3idx = Lidx[idx][2]

                        # product of wigner 3js
                        tmp[idx] *= self.pre_s[idx1, L1idx]
                        tmp[idx] *= self.pre_s[idx2, L2idx]
                        tmp[idx] *= self.pre_t[idx3, L3idx]

                        # sign (odd combinations are gone already)
                        tmp[idx] *= (-1)**(L1 + L2 + L3)
#                        print tmp

                    # reset tmp
                    tmp *= 0



    def fisher(self, comb='sstsst'):
        '''
        comb : str
            What combination of bispectra, e.g. sstsst
        '''

        DL1 = np.array([-1, 0, 1])
        DL2 = np.array([-1, 0, 1])
        DL3 = np.array([-2, -1, 0, 1, 2])

        for i, (l1, l2, l3) in enumerate(pc.ells):
            print idelta[i], (l1, l2, l3)
            Ls = pc.Ls[i]
            tri_cond = pc.tri_cond[i]

            n = 0
            print Ls[tri_cond].shape

            for j, (L1, L2, L3) in enumerate(Ls[tri_cond]):
                    for k, (L4, L5, L6) in enumerate(Ls[tri_cond]):

                        print L1, L2, L3, L4, L5, L6, n
                        n+= 1



if __name__ == '__main__':

    pass
