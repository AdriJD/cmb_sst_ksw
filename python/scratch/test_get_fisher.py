import numpy as np
from sst import Fisher

test_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20181025_sst/test'
camb_dir = '/mn/stornext/d8/ITA/spider/adri/analysis/20180911_sst/camb_output/lensed_r0_4000'

F = Fisher(test_dir)
F.get_camb_output(camb_out_dir=camb_dir)
radii = F.get_updated_radii()
radii = radii[::10]
F.get_bins(lmin=2, lmax=10, load=True, verbose=True)
F.get_beta(func='equilateral', load=True, verbose=True, radii=radii)
F.barrier()

if F.mpi_rank == 0:
    print F.bins['num_pass_full'].shape
    print F.bins['num_pass'].shape
else:
    import time
    time.sleep(3)
    print F.bins['num_pass'].shape
F.barrier()
exit()
F.get_binned_bispec('equilateral', load=False)

# bins needs to be recomputed when kwargs dont match
# beta needs to be recomputed when kwargs, radii, ks, transfers do not match
# bispec only needs to be recomputed when template doesnt match.

#beta = F.beta
#print beta
#beta_s = F.beta['beta_s']
#beta_t = F.beta['beta_t']
#print beta_s.shape
#print beta_t.shape
#print F.beta
#print F.bispec
