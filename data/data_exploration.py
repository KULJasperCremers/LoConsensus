import pickle
from pathlib import Path

import locomotif.visualize as visualize
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import SpectralClustering
from sklearn.manifold import MDS

motifs_file = Path('./data/motifs.pkl')
with motifs_file.open('rb') as f:
    motifs = pickle.load(f)

goffsets_file = Path('./data/goffsets.pkl')
with goffsets_file.open('rb') as f:
    goffsets = pickle.load(f)

series_file = Path('./data/series.pkl')
with series_file.open('rb') as f:
    series = pickle.load(f)

mvis = True
if mvis:
    fig, axs = visualize.plot_motif_sets(series, motifs)
    plt.savefig('./plots/motifs.png')


def find_timeseries_index(gindex, goffsets):
    for i in range(len(goffsets) - 1):
        if goffsets[i] <= gindex < goffsets[i + 1]:
            return i


n = len(goffsets) - 1
sm = np.zeros((n, n))
cm = np.zeros((n, n))

# yield (b, e), motif_set, csums, _
for mt in motifs:
    cs, ce = mt[0]
    cindex = find_timeseries_index(cs, goffsets)
    for i, motif in enumerate(mt[1]):
        ms, me = motif
        if cs == ms and ce == me:
            continue

        csim = mt[2][i]
        mindex = find_timeseries_index(ms, goffsets)
        sm[mindex][cindex] += csim
        sm[cindex][mindex] += csim
        cm[mindex][cindex] += 1
        cm[cindex][mindex] += 1

cm[cm == 0] = 1

sm = sm / cm
smin = np.percentile(sm, 5)
smax = np.percentile(sm, 95)
nm = np.clip((sm - smin) / (smax - smin), 0, 1)

print(f'nm: {nm.shape}')
print(nm)

dm = 1 - nm
print(f'dm: {dm.shape}')
Z = linkage(dm, method='average', optimal_ordering=True)

# labels = ['w1', 'r1', 'c1', 'w2', 'c2', 'w3']
labels = ['w1', 'r1', 'c1', 'w2', 'c2', 'w3', 'w11', 'r11', 'c11', 'w21', 'c21', 'w31']
plt.figure(figsize=(25, len(labels) * 3))
dendrogram(Z, labels=labels)
plt.savefig('./plots/dendogram.png')
plt.close()


spectral = SpectralClustering(
    n_clusters=3, affinity='precomputed', assign_labels='discretize'
)
clusters = spectral.fit_predict(nm)
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
points = mds.fit_transform(dm)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(points[:, 0], points[:, 1], c=clusters)
for i, label in enumerate(labels):
    plt.annotate(label, (points[i, 0], points[i, 1]))
plt.colorbar(scatter)
plt.savefig('./plots/scatter.png')
plt.close()
