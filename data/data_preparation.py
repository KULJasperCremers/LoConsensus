import pickle
import sys
from pathlib import Path

from scipy.interpolate import CubicSpline
from scipy.signal import decimate

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

import numpy as np
import pandas as pd
from constants import SAMPLING_FREQUENCY
from labels import LABELS


def get_activity_segments(labels, transitions):
    segments = []
    start_index = 0
    for transition in transitions:
        end_index = transition
        current_label = labels[start_index]
        segments.append((start_index, end_index, int(current_label)))
        start_index = end_index
    segments.append((start_index, len(labels), labels[start_index]))
    return segments


def interpolate(ts):
    nans = np.isnan(ts)
    if not np.any(nans):
        return ts

    nnans = ~nans
    indices = np.arange(len(ts))
    x = indices[nnans]
    y = ts[nnans]

    cspline = CubicSpline(x, y)
    interpolated_values = cspline(indices[nans])
    ts[nans] = interpolated_values
    return ts


def downsample_decimate(ts):
    decimated_data = decimate(ts, q=SAMPLING_FREQUENCY)
    return decimated_data


def is_znormalized(ts, tolerance=0.01):
    assert ts.ndim == 1 or ts.ndim == 2
    mean = np.mean(ts, axis=None)
    std = np.std(ts, axis=None)

    znormalized = (np.abs(mean) < tolerance) and (np.abs(std - 1) < tolerance)
    if ts.ndim == 2 and not znormalized:
        _, ndim = ts.shape
        for d in range(ndim):
            if not is_znormalized(ts[:, d]):
                return False
        return True

    return znormalized


def z_normalize(ts):
    return (ts - np.mean(ts, axis=None)) / np.std(ts, axis=None)


data_dir = Path('./data/Protocol')
data = {}
subject_files = data_dir.glob('subject*.dat')

for subject_file in subject_files:
    subject_id = subject_file.stem
    df = pd.read_csv(subject_file, sep=' ', header=None)
    labels = df.values[:, 1]
    # TODO: double check the colum indices here!
    # hand_x = df.values[:, 4]
    # hand_y = df.values[:, 5]
    # hand_z = df.values[:, 6]
    chest_x = df.values[:, 21]
    chest_y = df.values[:, 22]
    chest_z = df.values[:, 23]
    # TODO: double check the column indices here!
    # ankle_y = df.values[:, 38]
    # ankle_z = df.values[:, 39]
    # ankle_z = df.values[:, 40]

    transitions = np.where(np.diff(labels))[0] + 1

    activities = {
        'walking': 4,
        'cycling': 6,
        'running': 5,
    }
    activity_segments = get_activity_segments(labels, transitions)
    subject_segments = {activity: np.ndarray((0, 0)) for activity in activities}
    for start, end, segment_label in activity_segments:
        if (end - start) == 1:
            continue
        for activity, activity_label in activities.items():
            if segment_label == activity_label:
                chest_x_proc = z_normalize(
                    downsample_decimate(interpolate(chest_x[start:end]))
                )
                chest_y_proc = z_normalize(
                    downsample_decimate(interpolate(chest_y[start:end]))
                )
                chest_z_proc = z_normalize(
                    downsample_decimate(interpolate(chest_z[start:end]))
                )
                segment_data = np.column_stack(
                    (chest_x_proc, chest_y_proc, chest_z_proc)
                )
                subject_segments[activity] = segment_data
                break

    data[subject_id] = subject_segments

data_file = Path('./data/subjects.pkl')
with data_file.open('wb') as f:
    pickle.dump(data, f)
