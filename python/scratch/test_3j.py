import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0,'./../wigxjpf-1.6/pywigxjpf')
import pywigxjpf as wig
import os

opj = os.path.join
outdir = '/mn/stornext/d8/ITA/spider/adri/analysis/20180108_test_spin'

# Initialize
wig.wig_table_init(2*200,3)
wig.wig_temp_init(2*200)

# Note that arguments are in two_j = 2*j.
#val3j = wig.wig3jj([2* 10 , 2* 15 , 2* 10 ,\
#                    2*(-3), 2* 12 , 2*(-9)])
#print '3J(10  15  10; -3  12  -9):', val3j

m = 1
lmax = 100
trans = np.zeros((lmax+1, lmax+1), dtype=float)

def i_ell(ell):
    ret = np.sqrt( (2 * ell + 1) / 4. / np.pi)
    ret *= 4 * (-1)**ell
    ret /= float((ell * (ell + 1)))
    return ret
n=0
for lidx, ell in enumerate(xrange(2, lmax+1)):
    print ell
    for lpidx, ellp in enumerate(xrange(2, lmax+1)):
        n+=1
        for ellpp in xrange(np.abs(ell-ellp), ell+ellp + 1):

            if ellpp < 2:
                continue

            val = np.sqrt(2*ellpp+1) * i_ell(ellpp)
#            print val
            val *= wig.wig3jj([2*ell, 2*ellp, 2*ellpp,
                               2*-m, 2*(m+2), 2*-2])
#            print val
            val *= wig.wig3jj([2*ell, 2*ellp, 2*ellpp,
                               2*2, 2*0, 2*-2])
#            print val
            trans[lidx, lpidx] += val

        trans[lidx, lpidx] *= np.sqrt(2 * ellp + 1)

    trans[lidx, :] *= np.sqrt( (2 * ell + 1) / 4. / np.pi)
trans *= 2*np.pi
trans *= (-1)**m

# Free memory space
wig.wig_temp_free()
wig.wig_table_free()

plt.figure()
plt.imshow(trans, origin='lower')
plt.colorbar()
plt.savefig(opj(outdir, 'trans_lmax{}_m{}.png'.format(lmax, m)))
plt.close()


print trans
