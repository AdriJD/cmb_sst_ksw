'''

'''

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from scipy.special import spherical_jn
from scipy.interpolate import interp2d, griddata
sys.path.insert(0,'./../')
import fisher

opj = os.path.join
ana_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20171217_sst/'

test_dir = opj(ana_dir, 'bispectrum/test')

B = np.load(opj(test_dir, 'test.npy'))
bins = np.load(opj(test_dir, 'bins.npy'))

print B.shape
print bins

ell = np.arange(40, 200)

plt.figure()
for i in xrange(10):
    plt.plot(bins, B[0,i,:,0], '.')
plt.savefig(opj(ana_dir, 'bispectrum/test', 'test_1d.png'))
plt.close()


exit()


ell11, ell22 = np.meshgrid(ell, ell)
print ell11.shape
print ell22.shape

bins11, bins22 = np.meshgrid(bins, bins)
print bins11.shape
print B[0,:,:,0].shape
# First interpolate B to fill even sum(ell) gaps

#f = interp2d(bins, bins, B[0,:,:,0], kind='linear', fill_value=0.)
f = griddata((bins, bins), B[0,:,:,0], (ell11, ell22), method='nearest', fill_value=0.)

#Bi = f(ell, ell)
Bi = f

print Bi[40, 40]

plt.figure()
plt.imshow(Bi)
plt.colorbar()
plt.savefig(opj(ana_dir, 'bispectrum/test', 'test_interp.png'))
plt.close()

plt.figure()
plt.plot(Bi[10])
plt.savefig(opj(ana_dir, 'bispectrum/test', 'test_interp_1d.png'))
plt.close()

