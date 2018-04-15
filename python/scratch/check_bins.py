'''
Test binning
'''

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from scipy.special import spherical_jn
sys.path.insert(0,'./../')
import fisher

opj = os.path.join

ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/beta/transfer'

F = fisher.Fisher(
    camb_dir='/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/high_acy/nolens/')
#F = fisher.Fisher(camb_dir='/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/')
#F.get_camb_output(tag='test')
F.get_camb_output(tag='no_lens', lensed=False)
radii = F.get_updated_radii()

tr = F.depo['tensor']['transfer']
k = F.depo['tensor']['k']
lmax = F.depo['tensor']['lmax']

print tr.shape
print lmax
print k.size
print radii.size

F.init_bins()

print F.unique_ell
exit()
#print F.num_pass[80:81, 80:90, 80:90]
print F.num_pass[100:101, 100:101, 100:110]
#print F.num_pass.shape
#print F.first_pass.shape
#print F.bins
print F.unique_ell
print F.bins

f = F.local()
F.beta(f, 'scalar')
print F.depo['scalar']['beta']
print F.depo['scalar']['beta'].shape

np.save(opj('/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/camb_output/high_acy/beta_full',
            'beta_unique.npy'), F.depo['scalar']['beta'])
