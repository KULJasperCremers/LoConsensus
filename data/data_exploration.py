import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

motifs_file = Path('./data/motifs.pkl')
with motifs_file.open('rb') as f:
    motifs = pickle.load(f)

goffsets_file = Path('./data/goffsets.pkl')
with goffsets_file.open('rb') as f:
    goffsets = pickle.load(f)


def find_timeseries_index(gindex, goffsets):
    for i in range(len(goffsets) - 1):
        if goffsets[i] <= gindex < goffsets[i + 1]:
            return i


n = len(goffsets) - 1
sm = np.zeros((n, n))

# yield (b, e), motif_set, csums, _
for mt in motifs:
    cs, _ = mt[0]
    cindex = find_timeseries_index(cs, goffsets)
    for i, motif in enumerate(mt[1]):
        ms, _ = motif
        csim = mt[2][i]
        mindex = find_timeseries_index(ms, goffsets)
        sm[cindex][mindex] = csim

smin = sm.min()
smax = sm.max()
nm = (sm - smin) / (smax - smin)

d = 1 - nm
Z = linkage(d, method='average')

labels = ['subject105', 'subject101', 'subject106', 'subject102']
plt.figure()
dendrogram(Z, labels=labels)
plt.savefig('./plots/dendogram.png')
plt.close()
