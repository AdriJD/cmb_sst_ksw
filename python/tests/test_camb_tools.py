import unittest
import numpy as np
import os
from sst import camb_tools

opj = os.path.join

class TestTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_run_camb(self):
        
        lmax = 10
        k_eta_fac = 1
        transfer, cls, opts = camb_tools.run_camb(lmax, k_eta_fac=k_eta_fac,
                AccuracyBoost=1, lSampleBoost=1, lAccuracyBoost=1)

        # Check if output has expected shapes.
        ells_transfer = transfer['ells']
        k = transfer['k']
        exp_shape = (3, ells_transfer.size, k.size)
        self.assertEqual(transfer['scalar'].shape, exp_shape)

        ells_cls = cls['ells']
        spec_types = ['total', 'lens_potential', 'lensed_scalar', 
                      'unlensed_scalar', 'unlensed_total', 'tensor']
        
        for key in spec_types:
            if key == 'lens_potential':
                exp_shape = (3, ells_cls.size)
            else:    
                # II, EE, BB, TE
                exp_shape = (4, ells_cls.size)

            self.assertEqual(cls['cls'][key].shape, exp_shape)
        
        # I expect that Cls are in uK, also not Dells.

        # Not Dell. (Dell at ell=202 is ~5000 uK^2.
        # Note lmax is always around 300 at the least.
        self.assertTrue(cls['cls']['total'][0,150] < 100)

        # Not Kelvin.
        self.assertTrue(cls['cls']['total'][0,150] > 1e-3)

        # I expect that r = 1, so tensor BB at ell=80 != 0.
        bb_at_bump = cls['cls']['tensor'][2,78]
        self.assertTrue(bb_at_bump != 0.)

        # I expect that lensing is on, so total BB is larger than tensor-only.
        self.assertTrue(cls['cls']['total'][2,78] > bb_at_bump)
        
        # Check some important options.
        self.assertEqual(opts['cosmo']['TCMB'], 2.7255)
        self.assertEqual(opts['prim']['As'], 2.1056e-9)
        
