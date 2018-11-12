import numpy as np
from sst import Fisher

test_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20181112_sst/'
camb_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/high_acy/sparse_5000'

lmin = 2
lmax = 4999

F = Fisher(test_dir)
camb_opts = dict(camb_out_dir=camb_dir,
                 tag='r0',
                 lensed=False,
                 high_ell=True)

F.get_camb_output(**camb_opts)
radii = F.get_updated_radii()
radii = radii[::2]
F.get_bins(lmin=lmin, lmax=lmax, load=True, verbose=True, parity='odd')
F.get_beta(func='equilateral', load=False, verbose=True, radii=radii)
F.get_binned_bispec('equilateral', load=False)

    
    
