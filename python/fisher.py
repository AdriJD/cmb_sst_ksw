'''
Calculate elements of a Fisher matrix for a bispectrum analysis, 
add optimal estimator later on
'''

import sys
import os
import numpy as np
from scipy.special import spherical_jn
from scipy.integrate import trapz
import camb_tools as ct
import pywigxjpf as wig

opj = os.path.join

class PreCalc(object):
    
    def __init__(self, camb_dir='.'):
        '''        

        Keyword arguments
        -----------------
        camb_dir : str
            Path to directory containing camb output
            (Cls, transfers and aux) (default : ".")
        '''

        self.camb_dir = camb_dir

    def get_camb_output(self, **kwargs):
        '''
        Store CAMB ouput in internal dictionaries

        kwargs : {get_spectra_opts}
        '''
        source_dir = self.camb_dir
                
        cls = ct.get_spectra(source_dir, **kwargs)
        self.cls = cls

        self.depo = {}

        for ttype in ['scalar', 'tensor']:

            tr, lmax, k = ct.read_camb_output(source_dir, ttype=ttype)

            self.depo[ttype] = {'transfer' : tr,
                                    'lmax' : lmax,
                                    'k' : k}

    def get_default_radii(self):
        '''
        Get the radii (in Mpc) used in table 1 of liquori 2007
        '''

        low = np.linspace(0, 9377, num=98, dtype=float, endpoint=False)
        re1 = np.linspace(9377, 10007, num=18, dtype=float, endpoint=False)
        re2 = np.linspace(10007, 12632, num=25, dtype=float, endpoint=False)
        rec = np.linspace(12632, 13682, num=300, dtype=float, endpoint=False)

        radii = np.concatenate((low, re1, re2, rec))
        
        return radii

    def beta(self, f, ttype, L_range, radii=None):
        '''
        Calculate beta_l,L(r) = 2/pi * \int k^2 dk f(k) j_L(kr) T_X,l^(Z)(k)
        
        Arguments
        ---------
        f : array-like
            Factor f(k) of (primordial) factorized shape function.
        ttype : str
            "scalar" or "tensor"
        L_range : array-like
            Possible deviations from ell, e.g. [-2, -1, 0, 1, 2]
        
        Keyword arguments
        -----------------
        radii : array-like
            Array with radii to compute. In units [Mpc], if None, 
            use default_radii (default : None)
            
        Returns
        -------
        beta : array-like
            beta_ell_L (r) array of shape (r.size, lmax+1, L.size)
        '''

        # you want to allow f to be of shape (nfact, 3) 

        if not radii:
            # use Liguori 2007 values
            radii = self.get_default_radii()

        k = self.depo[ttype]['k']
        if k.size != f.size:
            raise ValueError('f and k not compatible: {}, {}'.format(
                    f.size, k.size))
        
        lmax = self.depo[ttype]['lmax']
        L_range = np.asarray(L_range)

        # scale f by k^2
        k2 = k**2
        f_scaled = f * k2

        transfer = self.depo[ttype]['transfer']
        pols = ['I', 'E', 'B'] if ttype == 'tensor' else ['I', 'E']

        # change beta to incorporate all factorized terms
        beta = np.zeros((radii.size, lmax+1, L_range.size, len(pols)))

        jL = np.zeros((L_range.size, k.size))

        for ridx, radius in enumerate(radii[::10]):
            
            kr = k * radius

            print ridx
            for lidx, ell in enumerate(xrange(lmax+1)):

                for Lidx, L in enumerate(L_range):
                    L = ell + L
                    if L < 0:
                        continue
    
                    if lidx == 0:
                        # first pass, fill all
                        jL[Lidx] = spherical_jn(L, kr)
                    else:
                        # second pass only fill new row
                        if Lidx == L_range.size - 1:
                            jL[Lidx] = spherical_jn(L, kr)
                        else:
                            pass
                    for pidx, pol in enumerate(pols):
                        
                        # here you'd want to treat all the factorized terms the prim. template

                        b_int = trapz(transfer[pidx,ell,:] * f_scaled * jL[Lidx], k)
                        beta[ridx, ell, Lidx, pidx] = b_int

                # permute rows such that oldest row can be replaced next ell
                jL = np.roll(jL, -1, axis=0)

        beta *= (2 / np.pi)

        self.depo[ttype]['beta'] = beta
        return beta


# class 2
# sky and experimental details
# for now, fsky, cls, nls (beam), pol combinations
class Experiment(object):
    
    def __init__(self):

        pass

# class 3
# bispectra. Start with local model in TSS.
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

    def local(self, fnl=1):
        '''
        eq. 18 from Meerburg 2016 without the angular
        dependent parts and I = local
        '''
        
        amp = (2 * np.pi)**3 * 16 * np.pi**4 * self.scalar_amp * np.sqrt(self.r)
        amp *= fnl

        ks = self.depo['scalar']['k']
        kt = self.depo['tensor']['k']

        km3 = k**-3
        zeros = np.zeros(k.size)

        template = np.asarray([[km3, km3, zeros], [km3, zeros, km3], [zeros, km3,km3]])
        template *= amp
        
        return template
# class 3
# fisher matrix
# for now, brute force the fisher matrix in python

class Fisher(Bispectrum, Experiment):
    
    def __init__(self, **kwargs):
        
        super(Fisher, self).__init__(**kwargs)


    

pc = Fisher(camb_dir='/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output')
pc.get_camb_output(tag='test')

#print pc.cls
#print pc.depo['tensor']['k']
#print 'aap'
k = pc.depo['tensor']['k']
kt = pc.depo['scalar']['k']
beta = pc.beta(k, 'tensor', [-1, 0, 1])
#print beta
#print beta.shape
#print beta.nbytes

print k.size
print kt.size

print pc.local().shape
