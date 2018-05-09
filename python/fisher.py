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
from scipy.interpolate import CubicSpline
import camb_tools as ct
import pywigxjpf as wig
from beamconv import instrument as instr
import warnings
from mpi4py import MPI

opj = os.path.join

class PreCalc(instr.MPIBase):

    def __init__(self, **kwargs):
        '''

        Keyword arguments
        -----------------
        kwargs : {MPIBase_opts}
        '''

        # TODO what goes in the depo and what does not?
        self.depo = {}

        self.depo['init_bins'] = False

        # Extract kwargs for loading CAMB output, such that
        # they do not get passed to MPIBase

        # init MPIBase 
        super(PreCalc, self).__init__(**kwargs)

        # Initialize wigner tables for lmax=8000
        wig.wig_table_init(2 * 8000, 9)
        wig.wig_temp_init(2 * 8000)
        
    def get_camb_output(self, camb_out_dir='.', high_ell=False, **kwargs):
        '''
        Store CAMB ouput (transfer functions and Cls) and store
        in internal dictionaries. 

        Keyword Arguments
        -----------------
        camb_out_dir : str
            Path to folder containing CAMB output. 
            (default : '.')
        high_ell : boool
            If True, load up additional transfer functions for high
            ell (>4000) that use sparse samples in ell. Is combined
            into single transfer function.
        kwargs : {get_spectra_opts}

        Notes
        -----
        Transfer functions and Cls are not independent,
        so makes sense to load them together like this.
        '''
        source_dir = camb_out_dir

        lmax = None
        cls = None
        tr = None
        k = None
        ells = None

        # load spectra
        if self.mpi_rank == 0:

            cls, lmax = ct.get_spectra(source_dir, **kwargs)

        cls = self.broadcast_array(cls)
        lmax = self.broadcast(lmax)

        self.depo['cls'] = cls
        self.depo['cls_lmax'] = lmax

        # load transfer functions, k-arrays and lmax
        for ttype in ['scalar', 'tensor']:

            if self.mpi_rank == 0:
                tr, lmax, k = ct.read_camb_output(source_dir,
                                                  ttype=ttype)

                if high_ell:
                    tr_hl, lmax_hl, k_hl, ells_hl = ct.read_camb_output(
                        source_dir, ttype=ttype, high_ell=True)

#                    print k_hl
#                    print k

#                    print k_hl.shape
#                    print k.shape

                    # disable this for tensors
                    if ttype == 'tensor':
                        tr_hl *= 0.

                    # remove low ell from tr_hl
                    # remember, tr has shape (num_s, ells, ks)
                    lidx_min_hl = np.where(ells_hl > lmax)[0][0]

                    # create ells that combines low and high
                    ells = np.concatenate((np.arange(2, lmax+1),
                                           ells_hl[lidx_min_hl:]))
                    
                    # index where sparse part starts in full ells array
                    lidx_min = np.where(ells > lmax)[0][0] 
                    assert ells[lidx_min] == ells_hl[lidx_min_hl]

                    # allocate combined transfer
                    num_s = tr_hl.shape[0]
                    tr_full = np.zeros((num_s,ells.size,k.size))

                    # fill with old transfer
                    tr_full[:,:lmax-1,:] = tr

                    # interpolate high ell transfer to low ell ks
                    print 'interpolating high_ell transfer...'
                    for nsidx in xrange(num_s):

                        # only loop over sparse part of ells
                        for lidx, ell in enumerate(ells[lidx_min:]):

                            lidx_hl = lidx + lidx_min_hl # index to ells_hl
                            lidx_f = lidx + lidx_min # index to ells
                            assert ells[lidx_f] == ells_hl[lidx_hl]

                            # cubic interpolation over k samples
                            cs = CubicSpline(k_hl, tr_hl[nsidx,lidx_hl,:])
                            tr_full[nsidx,lidx_f,:] = cs(k)

                    tr = tr_full
                    lmax = lmax_hl
                    print '...done'

            tr = self.broadcast_array(tr)
            k = self.broadcast_array(k)
            lmax = self.broadcast(lmax)

            if high_ell:
                ells = self.broadcast_array(ells)

            self.depo[ttype] = {'transfer' : tr,
                                'lmax' : lmax,
                                'k' : k,
                                'ells_sparse' : ells} # None normally

        # make sure tensor and scalar ks are equal (not 
        # guaranteed by CAMB). Needed b/c both T and S 
        # transfer functions are integrated over same k
        # when calculating beta
        ks = self.depo['scalar']['k']
        kt = self.depo['tensor']['k']

        np.testing.assert_allclose(ks, kt)        

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
        Get the radii (in Mpc) that are more suitable to post-Planck LCDM
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
        pass

    def get_default_bins(self):
        '''
        Return default bins 

        Returns
        -------
        bins : array-like
            Left side of default bins from ell=2 to ell=1e4
        '''
        # bins used in Meerburg 2016
#        bins_0 = np.arange(2, 101, 1)
        bins_0 = np.arange(2, 151, 1)        
#        bins_1 = np.arange(110, 510, 10)
        bins_1a = np.arange(153, 203, 3)
        bins_1b = np.arange(206, 506, 6)
#        bins_1c = np.arange(165, 500, 10)
#        bins_1 = np.arange(160, 510, 10)
        bins_2 = np.arange(510, 2010, 10)
        bins_3 = np.arange(2020, 10000, 20)
#        bins_3 = np.arange(2020, 10000, 40)
        bins = np.concatenate((bins_0, bins_1a, bins_1b,
                               bins_2, bins_3))
#        bins = np.concatenate((bins_0, bins_1,
#                               bins_2, bins_3))

        return bins
        
    def combine(self, a, b, c):
        '''
        Combine three int arrays into a single in array
        by packing them bitwise. Only works for
        non-negative ints < 16384
        '''

        # do some checks because this is a scary function
        a = np.asarray(a)
        b = np.asarray(b)
        c = np.asarray(c)

        if a.dtype != np.int64:
            raise ValueError('a')

        if b.dtype != np.int64:
            raise ValueError('b')

        if c.dtype != np.int64:
            raise ValueError('c')
        
        if np.max(a) >= 16384 or np.any(a < 0):
            raise ValueError('a')

        if np.max(b) >= 16384 or np.any(b < 0):
            raise ValueError('b')

        if np.max(c) >= 16384 or np.any(c < 0):
            raise ValueError('c')

        # shift a 28 bits to the left, b 14
        return (a << 28) | (b << 14) | c

    def unpack(self, comb):
        '''
        Unpack int array into 3 arrays

        comb: int, array-like
        '''        

        comb = np.asarray(comb)
        if comb.dtype != np.int64:
            raise ValueError('d')

        # shift a 28 bits back to right
        a = comb >> 28
        # shift b 14 bits back to right 
        # and set all above 16383 to zero (i.e. a)
        b = (comb >> 14) & 16383
        # set all above 16383 to zero (i.e. a and b)
        c = (comb & 16383)

        return a, b, c

    def init_bins(self, lmin=None, lmax=None,
                  bins=None, parity='odd',
                  verbose=False):
        '''
        Create default bins in ell-space. Stores attributes for
        unique_ells, bins, num_pass and first_pass

        Keyword Arguments
        -----------------
        lmin : int
            Value of first bin
        lmax : int
             Create bins up to (but not including lmax)
        bins : array-like
            Lower value of bins in ell. Must be monotonically increasing.
            No repeated values allowed. May be truncated depending on lmin
            lmax. if lmin and lmax are given, bins must encompass them.
        parity : str
            Consider tuples in ell that are parity "odd"
            or "even". If None, use both. (default : "odd")

        Notes
        -----
        Sum over ell is l1 <= l2 <= l3

        Stores following internal attributes

        lmin : int
        lmax : int
        ells : array-like
            Unbinned ell values from lmin to lmax
        bins : array-like
            Left (lower) side of bins used in rest of code
        num_pass : array-like
            number of tuples per 3d bin that pass triangle
            condition. 
            SIGMAi1i2i3 in bucher 2015.
            Shape : (nbins, nbins, nbins).
        first_pass : array-like
            Lowest sum tuple per 3d bin that passes triangle
            condition. Shape : (nbins, nbins, nbins, 3).
        unique_ells : array-like
            Values of ell that appear in first_pass. Used to only
            precompute Wigner 3j's for values that are used.
        '''

        # store parity
        self.depo['parity'] = parity

        # If needed, extract lmin, lmax from provided bins
        if bins is not None:
            bins = np.array(bins)
            if not lmin:
                lmin = bins[0]
            elif lmin < bins[0]:
                raise ValueError("lmin < mininum bin")
            if not lmax:
                lmax = bins[-1]

        elif not (lmin and lmax):
            raise ValueError('Provide lmin and lmax if bins=None')

        if lmax <= lmin:
            raise ValueError('lmax <= lmin')

        self.lmax = lmax
        self.lmin = lmin
        ells = np.arange(lmin, lmax+1)
        self.ells = ells

        if bins is None:
            bins = self.get_default_bins()

        # Truncate bins
        try:
            # find first occurences
            min_bidx = np.where(bins >= lmin)[0][0]
            max_bidx = np.where(bins > lmax)[0][0]
        except IndexError:
            if lmax == bins[-1]:
                max_bidx = None
            else:
                raise ValueError('lmin or lmax is larger than max bin')

        # note: we include bin that contains lmax, num_pass takes care of 
        # partial bin.
        bins = bins[min_bidx:max_bidx]

        # check for monotonic order and no repeating values.
        np.testing.assert_array_equal(np.unique(bins), bins)

        self.bins = bins
        num_bins = bins.size

        # allocate space for number of good tuples per 3d bin
#        # use unsigned int16 because these arrays can get really big
        num_pass = np.zeros((num_bins, num_bins, num_bins),
                            dtype=int)
        # allocate space for first good tuple per 3d bin
        first_pass = np.zeros((num_bins, num_bins, num_bins, 3), 
                              dtype=int)

        if parity == 'odd':
            pmod = 1
        elif parity == 'even':
            pmod = 0
        else:
            pmod = None

        # calculate bins in parallel, split ell in outer loop
        ells_sub = ells[self.mpi_rank::self.mpi_size]

        # create an ell -> bin idx lookup table for 2nd loop
        idx = np.digitize(ells, bins, right=False) - 1
        # boolean index array for inner loop determining good ell3
        ba = np.ones(ells.size, dtype=bool)

        if self.mpi_rank == 0 and verbose:
            print 'initializing bins...'

        for lidx1, ell1 in enumerate(ells_sub):
            if self.mpi_rank == 0 and verbose:
                print 'rank: {}, {}/{}'.format(self.mpi_rank,
                                               lidx1+1,ells_sub.size)
            # which bin?
            idx1 = idx[ell1 - lmin]

            for ell2 in xrange(ell1, lmax+1):

                # which bin?
                idx2 = idx[ell2 - lmin]

                # exclude ells below ell2
                ba[:ell2-lmin] = False

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
                bin_idx = idx[ba] # for each good ell, index of corresponding bin
                # find position in bin_idx of first unique index
                u_idx, first_idx, count = np.unique(bin_idx, return_index=True,
                                          return_counts=True)

                num_pass[idx1,idx2,u_idx] += count # count and u_idx have same size
                # first pass needs a (u_idx.size, 3) array
                # NOTE, this is weird, several lidx1, lidx2 may fall in 
                # same bin, you only use the last ell1, ell2 ?
                first_pass[idx1,idx2,u_idx,0] = ell1
                first_pass[idx1,idx2,u_idx,1] = ell2
                first_pass[idx1,idx2,u_idx,2] = gd_ells[first_idx]

                # reset array
                ba[:] = True

        # now combine num_pass and first_pass on root
        if self.mpi:
            self.barrier()

            method1 = True
            method2 = False

            if method1:
                # use mpi allreduce to sum num_pass
                # and take the min value of first_pass (modified to have no zeros)
                
                if self.mpi_rank == 0 and verbose:
                    print '...done, gathering bins from all ranks...'                

                first_pass[first_pass == 0] = 16383

                rec_num_pass = np.empty_like(num_pass)
                rec_first_pass = np.empty_like(num_pass) # note num instead of first

                # pack first pass bitwise
                # this works because ells are nonnegative and relatively
                # small
                first_pass_packed = self.combine(first_pass[:,:,:,0],
                                                 first_pass[:,:,:,1],
                                                 first_pass[:,:,:,2])
                
                self._comm.Allreduce(num_pass, rec_num_pass, op=MPI.SUM)
                # also use allreduce but only take minimum value of packed array
                # works because l1 <= l2 <= l3, so it correctly picks out the 
                # lowest sum tuple (
                self._comm.Allreduce(first_pass_packed, rec_first_pass,
                                     op=MPI.MIN)
                # unpack 
                rec_first_pass = np.stack(self.unpack(rec_first_pass), axis=3)
                rec_first_pass[rec_first_pass == 16383] = 0 # is that what i did before?

                self.barrier()

                num_pass = rec_num_pass
                first_pass = rec_first_pass

            if method2:

                if self.mpi_rank == 0 and verbose:
                    print '...done, gathering bins from all ranks...'

                # allocate space to receive arrays on root
                if self.mpi_rank == 0:
                    num_pass_rec = np.empty_like(num_pass)
                    first_pass_rec = np.empty_like(first_pass)

                # loop over all non-root ranks
                for rank in xrange(1, self.mpi_size):

                    # send arrays to root
                    if self.mpi_rank == rank:
                        self._comm.Send(num_pass, dest=0, tag=rank)
                        self._comm.Send(first_pass, dest=0, tag=rank + self.mpi_size)
                        print self.mpi_rank, 'sent'

                    if self.mpi_rank == 0:
                        self._comm.Recv(num_pass_rec, source=rank, tag=rank)
                        self._comm.Recv(first_pass_rec, source=rank, tag=rank + self.mpi_size)
                        print self.mpi_rank, 'root received'

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
        self.num_pass = num_pass
        self.first_pass = first_pass

        self.depo['init_bins'] = True

        if self.mpi_rank == 0 and verbose:
            print '...done'

    def get_binned_invcov(self, bins=None, ells=None, nls_tot=None):
        '''
        Combine signal and noise into an inverse cov matrix
        per bin. We first bin, then compute inverse.

        Keyword Arguments
        -----------------
        bins : array-like            
        ells : array-like
            lmax >= bins[-1]
        nls_tot : array-like
            Total covariance (Cl + Nl). Shape: (6, ells.size) in 
            order: TT, EE, BB, TE, TB, EB

        Notes
        -----
        Stores shape = (ells, 3, 3) array. ells is unbinned.
        '''

        if ells is None:
            ells = self.ells

        if bins is None:
            bins = self.bins        

        if not np.array_equal(ells, np.unique(ells)):
            raise ValueError('ells not monotonically increasing with Dl=1')

        if not ells[-1] >= bins[-1]:
            raise ValueError('lmax smaller than max bin')

        indices = np.digitize(ells, bins, right=False) - 1

        lmin = ells[0]
        lmax = ells[-1]

        if nls_tot is None:
            
            # get signal and noise from depo
            # eventually remove this probably
            cls = self.depo['cls'] # Signal, lmin = 2
            nls = self.depo['nls'] # Noise

            # to not mess up nls with cls later on
            nls = nls.copy()
            cls = cls.copy()

            # assume nls start at lmin, cls at ell=2
            # NOTE, for new noise, noise also starts at ell=2 ?
            nls = nls[:,:(lmax - lmin + 1)]
            cls = cls[:,lmin-2:lmax-1] 

            # Add signal cov to noise (cls for TB and TE not present in cls)
            nls[:4,:] += cls
            nls = nls_tot

        bin_cov = np.ones((bins.size, 3, 3))
        bin_cov *= np.nan
        bin_invcov = bin_cov.copy()

        nls_dict = {'TT': 0, 'EE': 1, 'BB': 2, 'TE': 3,
                    'ET': 3, 'BT': 4, 'TB': 4, 'EB': 5,
                    'BE': 5}

        # float array with bins + (lmax + 0.1) for binned_stat
        # binned_stat output is one less than input bins size
        bins_ext = np.empty(bins.size + 1, dtype=float)
        bins_ext[:-1] = bins
        bins_ext[-1] = lmax + 0.1
        
        for pidx1, pol1 in enumerate(['T', 'E', 'B']):
            for pidx2, pol2 in enumerate(['T', 'E', 'B']):

                # Cl+Nl array
                nidx = nls_dict[pol1+pol2]
                nell = nls_tot[nidx,:]

                # Bin
                bin_cov[:,pidx1,pidx2], _, _ = binned_statistic(ells, nell,
                                                             statistic='mean',
                                                             bins=bins_ext)

        # Invert
        for bidx in xrange(bins.size):
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
        self.bin_invcov = bin_invcov
        self.nls = nls_tot

    def init_wig3j(self):
        '''
        Precompute I^000_lL1 and I^20-2_lL2 for all unique ells
        and \Delta L \in [-2, -1, 0, 1, 2]

        Store internally as wig_s and wig_t arrays with shape:
        (ells.size, (\Delta L = 5)). So full ells-sized array, although
        only values in unique_ells are calculated.
        '''

        u_ells = self.unique_ells
        ells = self.ells

        lmin = ells[0]

        wig_s = np.zeros((ells.size, 5))
        wig_t = np.zeros((ells.size, 5))

        for ell in u_ells:
            # ell is unbinned, so we can simply infer the indices
            lidx = ell - lmin

            for Lidx, DL in enumerate([-2, -1, 0, 1, 2]):

                L = ell + DL

                if DL == -1 or DL == 1:
                    # only here we need to fill the scalar factor
                    tmp = wig.wig3jj([2*ell, 2*L, 2,
                                      0, 0, 0])
                    tmp *= np.sqrt((2 * L + 1) * (2 * ell + 1) * 3)
                    tmp /= (2 * np.sqrt(np.pi))
                    wig_s[lidx,Lidx] = tmp

                tmp = wig.wig3jj([2*ell, 2*L, 4,
                                  4, 0, -4])
                tmp *= np.sqrt((2 * L + 1) * (2 * ell + 1) * 5)
                tmp /= (2 * np.sqrt(np.pi))

                wig_t[lidx,Lidx] = tmp

        self.wig_s = wig_s
        self.wig_t = wig_t

    def init_pol_triplets(self):
        '''
        Store polarization triples internally in (..,3) shaped array.

        For now, only triplets containing a single B-mode.

        Notes
        -----
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

    def beta(self, f=None, L_range=[-2, -1, 0, 1, 2], radii=None, bin=True,
             optimize=True, verbose=False):
        '''
        Calculate beta_l,L(r) = 2/pi * \int k^2 dk f(k) j_L(kr) T_X,l^(Z)(k).
        Vectorized. MPI'd by radius.

        Keyword Arguments
        -----------------
        f : array-like
            Factor f(k) of (primordial) factorized shape function.
            Can be of shape (nfact, 3, k.size) If None, use local
            (default : None)
        radii : array-like
            Array with radii to compute. In units [Mpc], if None,
            use default_radii (default : None)
        L_range : array-like, int
            Possible deviations from ell, e.g. [-2, -1, 0, 1, 2].
            Must be monotonically increasing with steps of 2. 
            (default : [0])
        bin : bool
            Bin the resulting beta in ell. (default : True)
        optimize : bool
            Do no calculate spherical bessel for kr << L (default : 
            True)
        verbose : bool
            Print progress (default : False)

        Returns
        -------
        beta : array-like
            beta_ell_L (r) array of shape (r.size, lmax+1, L.size)
        '''

        if not self.depo['init_bins']:
            raise ValueError('bins not initialized')

        if self.depo['scalar']['ells_sparse'] is not None:
            ells = self.depo['scalar']['ells_sparse']

        else:
            ells = self.ells

        L_range = np.asarray(L_range)
        if np.any(~(L_range == np.unique(L_range))):
            print "L_range: ", L_range
            raise ValueError("L_range is not monotonically increasing "+
                             "with steps of 1")
            
        if f is None:
            f, _ = self.local()

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
        # do a weird split for more even load balance, as larger r is slower
        radii_per_rank = []

        radii_sub = radii[self.mpi_rank::self.mpi_size]
        for rank in xrange(self.mpi_size):
            radii_per_rank.append(radii[rank::self.mpi_size])

        # beta scalar and tensor
        beta_s = np.zeros((ells.size,L_range.size,nfact,ks,
                           len(pols_s), radii_sub.size))
        beta_t = np.zeros((ells.size,L_range.size,nfact,ks,
                           len(pols_t), radii_sub.size))

        # allocate space for bessel functions
        jL = np.zeros((L_range.size, k.size))

        # an array with all possible L values
        ells_ext = np.arange(ells[-1] + L_range[-1] + 1)

        for ridx, radius in enumerate(radii_sub):

            kr = k * radius
            # Array that gives L -> kr idx mapping, i.e. in which kr sits ell
            kr_idx = np.digitize(ells_ext, bins=kr, right=True) 
            kr_idx[kr_idx == kr.size] = kr.size - 1 # fix last element

            if self.mpi_rank == 0 and verbose:
                print 'rank: {}, ridx: {}/{}, radius: {} Mpc'.format(
                    self.mpi_rank, ridx, radii_sub.size - 1, radius)

            ell_prev = ells[0] - 1
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

                    # If possible, reuse spherical bessels j_L from ell-1
                    # in case these ells are not full-sized, dont do it                    
                    if lidx == 0 or ell - 1 != ell_prev:
                        # first pass, or sparse ells: fill all
                        jL[Lidx,kmin_idx:] = spherical_jn(L, kr[kmin_idx:])
                    else:
                        # second pass only fill new row
                        if Lidx == L_range.size - 1:
                            jL[Lidx,:] *= 0.
                            jL[Lidx,kmin_idx:] = spherical_jn(L, kr[kmin_idx:])

                    # loop over T, E, B
                    for pidx, pol in enumerate(pols_t):

                        if pol != 'B':
                            # note: this assumes no B contribution to scalar
                            # i.e. no lensing Cl^BB
#                            tmp_s[kmin_idx:] = transfer_s[pidx,ell-2,kmin_idx:] #NOTE ell
                            tmp_s[kmin_idx:] = transfer_s[pidx,lidx,kmin_idx:] 
                            tmp_s[kmin_idx:] *= jL[Lidx,kmin_idx:]

#                        tmp_t[kmin_idx:] = transfer_t[pidx,ell-2,kmin_idx:] # NOTE ell
                        tmp_t[kmin_idx:] = transfer_t[pidx,lidx,kmin_idx:]
                        tmp_t[kmin_idx:] *= jL[Lidx,kmin_idx:]

                        for nidx in xrange(nfact):
                            for kidx in xrange(ks):

                                if pol != 'B':

                                    # scalars
                                    integrand_s = tmp_s[kmin_idx:] * \
                                        f[nidx,kidx,kmin_idx:]
                                    b_int_s = trapz(integrand_s, k[kmin_idx:])
                                    beta_s[lidx,Lidx,nidx,kidx,pidx,ridx] = b_int_s

                                # tensors
                                integrand_t = tmp_t[kmin_idx:] * \
                                    f[nidx,kidx,kmin_idx:]
                                b_int_t = trapz(integrand_t, k[kmin_idx:])
                                beta_t[lidx,Lidx,nidx,kidx,pidx,ridx] = b_int_t

                # permute rows such that oldest row can be replaced next ell
                # no harm doing this even when ells are sparse, see above
                jL = np.roll(jL, -1, axis=0)

                ell_prev = ell

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
                    
                    beta_s_sub = np.ones((ells.size,L_range.size,nfact,
                                          ks,len(pols_s),r_size))
                    beta_t_sub = np.ones((ells.size,L_range.size,nfact,
                                          ks,len(pols_t),r_size))

#                    beta_s_sub *= np.nan
#                    beta_t_sub *= np.nan

                # send beta_sub to root
                if self.mpi_rank == rank:                    
                    self._comm.Send(beta_s, dest=0, tag=rank)
                    self._comm.Send(beta_t, dest=0,
                                    tag=rank + self.mpi_size + 1)

                if self.mpi_rank == 0:
                    self._comm.Recv(beta_s_sub,
                                    source=rank, tag=rank)
                    self._comm.Recv(beta_t_sub, source=rank,
                                    tag=rank + self.mpi_size + 1)

                    # place into beta_full
                    for ridx, radius in enumerate(radii_per_rank[rank]):

                        # find radius index in total radii
                        ridx_tot, = np.where(radii == radius)[0]

                        beta_s_full[:,:,:,:,:,ridx_tot] = \
                            beta_s_sub[:,:,:,:,:,ridx]
                        beta_t_full[:,:,:,:,:,ridx_tot] = \
                            beta_t_sub[:,:,:,:,:,ridx]

            # broadcast full beta array to all ranks
            beta_s = self.broadcast_array(beta_s_full)
            beta_t = self.broadcast_array(beta_t_full)

        if not bin:
            self.depo['scalar']['beta'] = beta_s
            self.depo['tensor']['beta'] = beta_t
            return

        # Bin beta
        bins = self.bins
#        indices = np.digitize(ells, bins, right=False) - 1

        beta_s_f = np.asfortranarray(beta_s)
        beta_t_f = np.asfortranarray(beta_t)

        b_beta_s_f = np.zeros((bins.size,L_range.size,nfact,
                               ks,len(pols_s),radii.size))
        b_beta_t_f = np.zeros((bins.size,L_range.size,nfact,
                               ks,len(pols_t),radii.size))

        b_beta_s_f = np.asfortranarray(b_beta_s_f)
        b_beta_t_f = np.asfortranarray(b_beta_t_f)

        # float array with bins + (lmax + 0.1) for binned_stat
        # binned_stat output is one less than input bins size
        bins_ext = np.empty(bins.size + 1, dtype=float)
        bins_ext[:-1] = bins
        bins_ext[-1] = self.lmax + 0.1

        for pidx, pol in enumerate(pols_t):
            for kidx in xrange(ks):
                for ridx, radius in enumerate(radii):
                    for nidx in xrange(nfact):
                        for Lidx, L in enumerate(L_range):

                            if pol != 'B':
                                # scalar
                                tmp_beta = beta_s_f[:,Lidx,nidx,kidx,pidx,ridx]

                                b_beta_s_f[:,Lidx,nidx,kidx,pidx,ridx], _, _ = \
                                    binned_statistic(ells, tmp_beta, statistic='mean',
                                                     bins=bins_ext)

                                # Check for nans!
                                
                                    
                                # expand to full size
                                # Why not keep beta unbinned? 
#                                beta_s_f[:,Lidx,nidx,kidx,pidx,ridx] = \
#                                    b_beta_s_f[indices,Lidx,nidx,kidx,pidx,ridx]


                            # tensor
                            tmp_beta = beta_t_f[:,Lidx,nidx,kidx,pidx,ridx]

                            b_beta_t_f[:,Lidx,nidx,kidx,pidx,ridx], _, _ = \
                                binned_statistic(ells, tmp_beta, statistic='mean',
                                                 bins=bins_ext)

                            # Check for nans!

                            # expand to full size
#                            beta_t_f[:,Lidx,nidx,kidx,pidx,ridx] = \
#                                b_beta_t_f[indices,Lidx,nidx,kidx,pidx,ridx]

        # check for nans
        if np.any(np.isnan(beta_s)):
            raise ValueError('nan in beta_s')
        if np.any(np.isnan(beta_t)):
            raise ValueError('nan in beta_t')            

        beta_s = np.ascontiguousarray(beta_s_f)
        beta_t = np.ascontiguousarray(beta_t_f)

        b_beta_s = np.ascontiguousarray(b_beta_s_f)
        b_beta_t = np.ascontiguousarray(b_beta_t_f)

        # so now these are the unbinned versions
        # Note that they might be using a sparse high ell
        self.depo['scalar']['beta'] = beta_s
        self.depo['tensor']['beta'] = beta_t

        # binned versions
        self.depo['scalar']['b_beta'] = b_beta_s
        self.depo['tensor']['b_beta'] = b_beta_t

        return

class Template(PreCalc):
    '''
    Create arrays of shape (nfact, 3, k.size)
    '''

    def __init__(self, template='local', **kwargs):
        '''
     
        Keyword Arguments
        -----------------
        template : str
            Which template, either local, equilateral or orthogonal
        kwargs : {MPIBase_opts}

        Notes
        -----
        <h zeta zeta > = (2pi)^3 f(k1, k2, k3) delta(k1 + k2 + k3)
        * e(\hat(k1)) \hat(k2) \hat(k3),
        where f = 16 pi^4 As^2 fnl * S

        S (shape) is for (see Planck XVII):
        local: S = 2(1/(k1k2)^3 + cycl.),
        equil: S = 6(-1/(k1k2)^3 - cycl.  - 2/(k1k2k3)^2 + 1/(k1k2k3^3) )
        ortho: S = 6(-3/(k1k2)^3 - cycl.  - 8/(k1k2k3)^2 + 1/(k1k2k3^3) )

        For local we only need alpha and beta.
        For equilateral and orthgonal we need alpha, beta, gamma, delta 
        (see creminelli 2005).
        '''
        
        # You don't have to get primordial parameters from CAMB
        # transfer functions don't depend on them        
        self.scalar_amp = 2.1e-9
        # NOTE (2pi)^3????
#        self.common_amp = (2*np.pi)**3 * 16 * np.pi**4 * self.scalar_amp**2
        # this is the amplitude in e.g. f(k1, k2, k3) = amp * local
        self.common_amp = 16 * np.pi**4 * self.scalar_amp**2

        super(Template, self).__init__(**kwargs)

        # TODO: add attribute that tells you which template is used
        # load up template based on choice in init
        self.template = template

    def local(self, fnl=1):
        '''
        eq. 18 from Meerburg 2016 without the angular
        dependent parts and I = local

        Returns
        -------
        template : array-like
            shape (2, k.size) used for beta (0), alpha (1).
            For beta we simply have k^-3, for alpha
            just an array of ones.
        amp : float
            scalar amplitude of bispectrum with local
            shape: = 2 * fnl

        Notes
        -----        
        '''

        ks = self.depo['scalar']['k']

        km3 = ks**-3
        ones = np.ones(ks.size)

        template = np.asarray([km3, ones])
        amp = 2. * fnl

        return template, amp

    def equilateral(self, fnl=1):
        '''

        '''
        ks = self.depo['scalar']['k']

        km3 = ks**-3
        km2 = ks**-2
        km1 = ks**-1
        ones = np.ones(ks.size)

        template = np.asarray([km3, km2, km1, ones])
        amp = 6. * fnl
        
        return template, amp

    def orthogonal(self, fnl=1):
        '''
        Note, same as equil?
        '''
        ks = self.depo['scalar']['k']

        km3 = ks**-3
        km2 = ks**-2
        km1 = ks**-1
        ones = np.ones(ks.size)

        template = np.asarray([km3, km2, km1, ones])
        amp = 6. * fnl
        
        return template, amp

class Fisher(Template):

    def __init__(self, **kwargs):
        '''

        Keyword Arguments
        -----------------
        kwargs : {MPIBase_opts}
        
        '''

        super(Fisher, self).__init__(**kwargs)

    def binned_bispectrum(self, DL1, DL2, DL3, template='local'):
        '''
        Return binned bispectrum for given \Delta L triplet.

        Arguments
        ---------
        DL1 : Delta L_1
        DL2 : Delta L_2
        DL3 : Delta L_3

        Returns
        -------
        
        binned_bispectrum : array-like 
            Shape: (num_bins-1,num_bins-1,num_bins-1, pols)
        '''

        # loop over bins
        bins = self.bins
        ells = self.ells
        lmin = ells[0]
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
        
        # Indices to L arrays
        Lidx1 = DL1 + 2
        Lidx2 = DL2 + 2
        Lidx3 = DL3 + 2

        # define function names locally for faster lookup
        wig3jj = wig.wig3jj
        wig9jj = wig.wig9jj
        trapz_loc = trapz

        bins_outer_f = bins.copy() # f = full
#        # remove last bin
        idx_outer_f = np.arange(bins_outer_f.size)

        # scatter
        bins_outer = bins_outer_f[self.mpi_rank::self.mpi_size]
        idx_outer = idx_outer_f[self.mpi_rank::self.mpi_size]

        # make every rank know idx_outer on other ranks
        idx_per_rank = []
        for rank in xrange(self.mpi_size):
            idx_per_rank.append(idx_outer_f[rank::self.mpi_size])
            
        # allocate bispectrum
        nbins = bins.size
        bispectrum = np.zeros((bins_outer.size, nbins, nbins, psize))

        for idxb, (idx1, i1) in enumerate(zip(idx_outer, bins_outer)):
            # note, idxb is bins_outer index for bispectrum per rank
            # idx1 is index to full-sized bin array, i1 is bin
#            print self.mpi_rank, idxb, idx1
            # load binned beta
            beta_s_l1 = beta_s[idx1,Lidx1,0,0,:,:] # (2,r.size)
            beta_t_l1 = beta_t[idx1,Lidx1,0,0,:,:] # (3,r.size)

            alpha_s_l1 = beta_s[idx1,Lidx1,0,1,:,:] # (2,r.size)
            alpha_t_l1 = beta_t[idx1,Lidx1,0,1,:,:] # (3,r.size)


            for idx2, i2 in enumerate(bins[idx1:]):
                idx2 += idx1

                # load binned beta
                beta_s_l2 = beta_s[idx2,Lidx2,0,0,:,:] # (2,r.size)
                beta_t_l2 = beta_t[idx2,Lidx2,0,0,:,:] # (3,r.size)

                alpha_s_l2 = beta_s[idx2,Lidx2,0,1,:,:] # (2,r.size)
                alpha_t_l2 = beta_t[idx2,Lidx2,0,1,:,:] # (3,r.size)

                for idx3, i3 in enumerate(bins[idx2:]):
                    idx3 += idx2
                    
                    num = num_pass[idx1, idx2, idx3]
                    if num == 0.: 
                        # No valid ell tuples in this bin
                        # Note: do not impose coditions on 
                        # the bins here!
                        continue

                    # load binned beta
                    beta_s_l3 = beta_s[idx3,Lidx3,0,0,:,:] # (2,r.size)
                    beta_t_l3 = beta_t[idx3,Lidx3,0,0,:,:] # (3,r.size)

                    alpha_s_l3 = beta_s[idx3,Lidx3,0,1,:,:] # (2,r.size)
                    alpha_t_l3 = beta_t[idx3,Lidx3,0,1,:,:] # (3,r.size)

                    # Load the ell triplets used per bin
                    ell1, ell2, ell3 = first_pass[idx1, idx2, idx3,:]

                    if ell1 < 2 or ell2 < 2 or ell3 < 2:
                        raise ValueError("ells are wrong")                        

                    # indices to full-size ell arrays
                    lidx1 = ell1 - lmin
                    lidx2 = ell2 - lmin
                    lidx3 = ell3 - lmin

                    L1 = DL1 + ell1
                    L2 = DL2 + ell2
                    L3 = DL3 + ell3

                    # Calculate the angular prefactor (eq. 6.19 thesis Shiraishi)
                    # before polarization loop 
                    # Overall angular part 
                    ang = wig3jj([2*L1, 2*L2, 2*L3,
                                  0, 0, 0])

                    # NOTE: TODO, B must be imag, so check this
                    ang *= np.real((-1j)**(ell1 + ell2 + ell3 - 1)) 
                    ang *= (-1)**((L1 + L2 + L3)/2)
                
                    ang *= np.sqrt( (2*L1 + 1) * (2*L2 + 1) * (2*L3 + 1))
                    ang /= (2. * np.sqrt(np.pi))

                    # calculate the angular parts for each S, S, T comb
                    # TSS
                    ang_tss = wig_t[lidx1, Lidx1]
                    ang_tss *= wig_s[lidx2, Lidx2]
                    ang_tss *= wig_s[lidx3, Lidx3]

                    ang_tss *= ang
                    if ang_tss != 0.: 
                        ang_tss *= wig9jj( [(2*ell1),  (2*ell2),  (2*ell3),
                                                (2*L1),  (2*L2),  (2*L3),
                                                4,  2,  2] ) #NOTE HERE

                    else:
                        # don't waste time on 9j if zero anyway
                        pass

                    # STS
                    ang_sts = wig_s[lidx1, Lidx1]
                    ang_sts *= wig_t[lidx2, Lidx2]
                    ang_sts *= wig_s[lidx3, Lidx3]

                    ang_sts *= ang
                    if ang_sts != 0.:
                        ang_sts *= wig9jj( [(2*ell1),  (2*ell2),  (2*ell3),
                                                (2*L1),  (2*L2),  (2*L3),
                                                2, 4,  2] )

                    else:
                        pass
                    # TSS
                    ang_sst = wig_s[lidx1, Lidx1]
                    ang_sst *= wig_s[lidx2, Lidx2]
                    ang_sst *= wig_t[lidx3, Lidx3]

                    ang_sst *= ang
                    if ang_sst != 0.:
                        ang_sst *= wig9jj( [(2*ell1),  (2*ell2),  (2*ell3),
                                                (2*L1),  (2*L2),  (2*L3),
                                                2,  2,  4] )

                    else:
                        pass

                    if ang_tss == 0. and ang_sts == 0. and ang_sst == 0. and ell1!=ell2!=ell3:
                        # wrong L,ell comb, determine what went wrong
                        print ell1, ell2, ell3, L1, L2, L3, num
                        print ang_tss, ang_sts, ang_sst
                        print np.abs(ell1 - ell2) <= ell3 <= ell1 + ell2
                        print np.abs(L1 - L2) <= L3 <= L1 + L2
                        print np.abs(ell1 - L1) <= 2 <= ell1 + L1
                        print np.abs(ell1 - L1) <= 4 <= ell1 + L1
                        print np.abs(ell2 - L2) <= 2 <= ell2 + L2
                        print np.abs(ell2 - L2) <= 4 <= ell2 + L2
                        print np.abs(ell3 - L3) <= 2 <= ell3 + L3
                        print np.abs(ell3 - L3) <= 4 <= ell3 + L3
                        raise ValueError("angular prefactor is zero")
                        #continue

                    # loop over pol combs
                    # TODO put this in function
                    for pidx in xrange(psize):

                        pidx1, pidx2, pidx3 = pol_trpl[pidx,:]

                        # integrate over bba + bab + abb for each T, S, S comb
                        integrand[:] = 0.

                        # TSS
                        if pidx2 == 2 or pidx3 == 2 or ang_tss == 0.:
                            # no B-mode for scalar or this l, L combination
                            pass
                        else:
                            if pidx1 == 2:
                                assert (1 + L1 + ell1)%2 == 0

                            integrand_tss[:] = beta_t_l1[pidx1,:] * \
                                beta_s_l2[pidx2,:] * alpha_s_l3[pidx3,:]

                            integrand_tss += beta_t_l1[pidx1,:] * \
                                alpha_s_l2[pidx2,:] * beta_s_l3[pidx3,:]

                            integrand_tss += alpha_t_l1[pidx1,:] * \
                                beta_s_l2[pidx2,:] * beta_s_l3[pidx3,:]

                            integrand_tss *= ang_tss
                            
                            integrand += integrand_tss

                        # STS
                        if pidx1 == 2 or pidx3 == 2 or ang_sts == 0.:
                            # no B-mode for scalar
                            pass
                        else:
                            if pidx2 == 2:
                                assert (1 + L2 + ell2)%2 == 0

                            integrand_sts[:] = beta_s_l1[pidx1,:] * \
                                beta_t_l2[pidx2,:] * alpha_s_l3[pidx3,:]

                            integrand_sts += beta_s_l1[pidx1,:] * \
                                alpha_t_l2[pidx2,:] * beta_s_l3[pidx3,:]

                            integrand_sts += alpha_s_l1[pidx1,:] * \
                                beta_t_l2[pidx2,:] * beta_s_l3[pidx3,:]

                            integrand_sts *= ang_sts

                            integrand += integrand_sts

                        # SST
                        if pidx1 == 2 or pidx2 == 2 or ang_sst == 0.:
                            # no B-mode for scalar
                            pass
                        else:
                            if pidx3 == 2:
                                assert (1 + L3 + ell3)%2 == 0

                            integrand_sst[:] = beta_s_l1[pidx1,:] * \
                                beta_s_l2[pidx2,:] * alpha_t_l3[pidx3,:]

                            integrand_sst += beta_s_l1[pidx1,:] * \
                                alpha_s_l2[pidx2,:] * beta_t_l3[pidx3,:]

                            integrand_sst += alpha_s_l1[pidx1,:] * \
                                beta_s_l2[pidx2,:] * beta_t_l3[pidx3,:]

                            integrand_sst *= ang_sst

                            integrand += integrand_sst
                        
                        # Integrate over r
                        integrand *= r2
                        bispec = trapz_loc(integrand, radii)

                        # Multiply by num (note, already floats)
                        # Note that for plotting purposes, you need to remove
                        # the num factor again
                        # Note, in Fisher, num cancels with binned (Cl)^-1's
                        bispec *= num 
                        
                        bispectrum[idxb,idx2,idx3,pidx] = bispec             
            
        bispectrum *= (8 * np.pi)**(3/2.) / 3.

        # Each rank has, for a unordered set of i1, the i2, i3, pol bispectra 
        # Now we add them together on the root rank
        if self.mpi:
            self.barrier()

            # create empty full-sized bispectrum on root
            if self.mpi_rank == 0:

                bispec_full = np.zeros((bins_outer_f.size,nbins,nbins,psize))
                
                # place sub B on root in full B for root
                for i, fidx in enumerate(idx_per_rank[0]):
                    # i is index to sub, fidx index to full
                    bispec_full[fidx,...] = bispectrum[i,...]

            else:
                bispec_full = 0

            # loop over all non-root ranks
            for rank in xrange(1, self.mpi_size):

                if self.mpi_rank == 0:
                    # allocate B sub on root for receiving
                    bin_size = idx_per_rank[rank].size
                    bispec_rec = np.zeros((bin_size,nbins,nbins,psize))

                # send bispectrum to root
                if self.mpi_rank == rank:
                    self._comm.Send(bispectrum, dest=0, tag=rank)

                if self.mpi_rank == 0:
                    self._comm.Recv(bispec_rec, source=rank, tag=rank)

                    # place into root bispectrum                    
                    for i, fidx in enumerate(idx_per_rank[rank]):
                        # i is index to sub, fidx index to full
                        bispec_full[fidx,...] = bispec_rec[i,...]

            return bispec_full

        else:
            # no MPI, so process already has full bispectrum
            return bispectrum

