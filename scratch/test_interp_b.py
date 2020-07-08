import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import sys

import numpy as np
from scipy.interpolate import Rbf

# Create bispectrum
m = 4
n = 100
bins = np.arange(n)
bispec = np.random.randn(int(m * n ** 2))
bispec = bispec.reshape((m, n, n))
#bispec = bispec.reshape((n, n))

print bispec.size

x, y, z = np.mgrid[0:m, 0:n, 0:n]
#x, y = np.mgrid[0:n, 0:n]

def norm_numpy(x1, x2):
    return np.linalg.norm(x1 - x2, axis=0)

ret = Rbf(x, y, z, bispec, function='multiquadric', smooth=0)
#ret = Rbf(x, y, bispec, function='linear', smooth=0)

#x, y, z = np.mgrid[0:m, 0:5*n, 0:5*n]
#ret(x, y, z)

print ret
print sys.getsizeof(ret)
