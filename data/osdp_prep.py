import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from matplotlib import rcParams
from pandas import read_csv
from scipy.interpolate import CubicSpline
from scipy.stats import zscore

rcParams.update({'figure.autolayout': True})

# from kshape.core import kshape, zscore
# from dtaidistance import dtw

# from tqdm import tqdm
# from preprocessingandrecords import *
# from experimentsetup import *
# from plotty import *

import matplotlib.gridspec as gridspec
from scipy.cluster.hierarchy import dendrogram, linkage

main_dir = './data/OSPD/'
dataset = None
for root, dirs, files in os.walk(main_dir):
    for f in files:
        if '60min_singleindex.csv' in f:
            dataset = read_csv(main_dir + f)
            break

fullcountrylist = []
for k in dataset.keys():
    if 'load_actual_entsoe_transparency' in k:
        if '50hertz' not in k:
            fullcountrylist.append(k)


l = [
    'Austria',
    'Cyprus',
    'Germany',
    'Denmark',
    'Estonia',
    'Spain',
    'Great Britain',
    'United Kingdom',
    'Greece',
    'Croatia',
    'Hungary',
    'Italy',
    'Lithuania',
    'Latvia',
    'Norway',
    'Portugal',
    'Sweden',
    'Slovakia',
]

lab = [
    'AT_load_actual_entsoe_transparency',
    'CY_load_actual_entsoe_transparency',
    'DE_load_actual_entsoe_transparency',
    'DK_load_actual_entsoe_transparency',
    'EE_load_actual_entsoe_transparency',
    'ES_load_actual_entsoe_transparency',
    'GB_GBN_load_actual_entsoe_transparency',
    'GB_UKM_load_actual_entsoe_transparency',
    'GR_load_actual_entsoe_transparency',
    'HR_load_actual_entsoe_transparency',
    'HU_load_actual_entsoe_transparency',
    'IT_load_actual_entsoe_transparency',
    'LT_load_actual_entsoe_transparency',
    'LV_load_actual_entsoe_transparency',
    'NO_load_actual_entsoe_transparency',
    'PT_load_actual_entsoe_transparency',
    'SE_load_actual_entsoe_transparency',
    'SK_load_actual_entsoe_transparency',
]

cc = [
    'AT',
    'CY',
    'DE',
    'DK',
    'EE',
    'ES',
    'GBGBN',
    'GBUKM',
    'GR',
    'HR',
    'HU',
    'IT',
    'LT',
    'LV',
    'NO',
    'PT',
    'SE',
    'SK',
]

label_colors = {i: 'k' for i in l}

print(len(l), len(lab))

a = np.array(dataset['utc_timestamp'])
print('Timestamps: {}, {}'.format(a[0], a[-1]))
data_list = []
for lk_idx in range(len(lab)):
    lk = lab[lk_idx]
    ts = dataset[lk]
    data_list.append(ts)

print(len(data_list))


def interpolate(ts):
    ts_copy = ts.copy()
    nans = np.isnan(ts_copy)
    if not np.any(nans):
        return ts_copy

    nnans = ~nans
    indices = np.arange(len(ts_copy))
    x = indices[nnans]
    y = ts_copy[nnans]

    cspline = CubicSpline(x, y)
    interpolated_values = cspline(indices[nans])
    ts_copy[nans] = interpolated_values
    return ts_copy


def z_normalize(ts):
    return (ts - np.mean(ts, axis=None)) / np.std(ts, axis=None)


ts_list = []
for i, ts in enumerate(data_list):
    r = ts[4:1000]
    if np.any(np.isnan(r)):
        continue
    z = z_normalize(r)
    if ts.ndim == 1:
        e = np.expand_dims(z, axis=1)
    ts_list.append((np.array(e, dtype=np.float32), cc[i]))

print(len(ts_list))

data_file = Path('./data/osdp.pkl')
with data_file.open('wb') as f:
    pickle.dump(ts_list, f)
