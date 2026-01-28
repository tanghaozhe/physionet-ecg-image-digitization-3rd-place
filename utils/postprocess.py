import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.signal import savgol_filter
from scipy.signal import resample
from scipy.interpolate import PchipInterpolator
from scipy.signal import detrend



# 26.3986
# resample 27.8248？？？？？？
# resample + PchipInterpolator 27.8252
def pixel_to_series(pixel, zero_mv, length):
    if isinstance(pixel, np.ndarray):
        pixel = torch.from_numpy(pixel)
    
    C, H, W = pixel.shape
    
    vals_max, indices_max = torch.max(pixel, dim=1)
    
    indices_left = torch.clamp(indices_max - 1, min=0)
    indices_right = torch.clamp(indices_max + 1, max=H - 1)
    
    vals_left = torch.gather(pixel, 1, indices_left.unsqueeze(1)).squeeze(1)
    vals_right = torch.gather(pixel, 1, indices_right.unsqueeze(1)).squeeze(1)
    vals_center = vals_max
    
    denom = vals_left - 2 * vals_center + vals_right + 1e-8
    delta = 0.5 * (vals_left - vals_right) / denom
    
    delta = torch.clamp(delta, -0.5, 0.5)
    
    y_coords = indices_max.float() + delta
    
    confidence_probs = vals_center
    confidence_threshold = 0.15
    
    series = y_coords.cpu().numpy()
    conf_np = confidence_probs.cpu().numpy()
    
    for j in range(C):
        mask_miss = conf_np[j] < confidence_threshold
        if np.all(mask_miss):
            series[j][:] = zero_mv[j]
        elif np.any(mask_miss):
            valid_x = np.where(~mask_miss)[0]
            valid_y = series[j][valid_x]
            miss_x = np.where(mask_miss)[0]
            if len(valid_x) > 2:
                itp = PchipInterpolator(valid_x, valid_y)
                series[j][miss_x] = itp(miss_x)
            else:
                series[j][miss_x] = np.interp(miss_x, valid_x, valid_y)

    if length is not None and length != W:
        series = resample(series, int(length), axis=-1)


    return series.astype(np.float32)



def filter_series_by_limits(series, limits=[[-2, 2], [-2, 2], [-4, 4], [-4, 4]]):
    _, L = series.shape
    series[3] = np.clip(series[3], -1, 1)
    for j in [0, 1, 2]:
        for i in range(4):
            i0 = i * (L // 4)
            i1 = (i + 1) * (L // 4)
            series[j, i0:i1] = np.clip(series[j, i0:i1], *limits[i])

    return series

def split_to_lead(series, split_length):
    LEAD = [
        ['I', 'aVR', 'V1', 'V4'],
        ['II', 'aVL', 'V2', 'V5'],
        ['III', 'aVF', 'V3', 'V6'],
    ]
    index = np.cumsum(split_length)[:-1]

    lead = {}
    for i in range(3):
        split = np.split(series[i], index)
        for (k, s) in zip(LEAD[i], split):
            lead[k] = s
    lead['II-rhythm'] = series[3]
    return lead


def csv_to_kaggle_format(truth_csv, sample_id, fs):
    rows = []
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    for lead in leads:
        if lead not in truth_csv.columns:
            continue
        signal = truth_csv[lead].dropna().values
        for row_idx, value in enumerate(signal):
            rows.append({
                'id': f'{sample_id}_{row_idx}_{lead}',
                'fs': fs,
                'value': value
            })
    return rows


def dw(series_dict, alpha=0.33):
    if all(k in series_dict for k in ['I', 'II', 'III']):
        L1 = series_dict['I']
        L2 = series_dict['II']
        L3 = series_dict['III']

        error = L2 - (L1 + L3)

        series_dict['I'] = L1 + (alpha * error)
        series_dict['III'] = L3 + (alpha * error)
        series_dict['II'] = L2 - (alpha * error)
    
    return series_dict

def dw_after_weighted(series_dict, weights={'I': 1.0, 'II': 2.0, 'III': 1.0}):
    if all(k in series_dict for k in ['I', 'II', 'III']):
        len_I = len(series_dict['I'])
        len_II = len(series_dict['II'])
        len_III = len(series_dict['III'])
        n = min(len_I, len_II, len_III)

        part_I = series_dict['I'][:n]
        part_II = series_dict['II'][:n]
        part_III = series_dict['III'][:n]

        error = part_II - (part_I + part_III)

        w_I = weights.get('I', 1.0)
        w_II = weights.get('II', 1.0)
        w_III = weights.get('III', 1.0)

        inv_I = 1.0 / w_I
        inv_II = 1.0 / w_II
        inv_III = 1.0 / w_III
        
        total_inv_weight = inv_I + inv_II + inv_III

        ratio_I = inv_I / total_inv_weight
        ratio_II = inv_II / total_inv_weight
        ratio_III = inv_III / total_inv_weight

        corr_I = error * ratio_I
        corr_III = error * ratio_III
        corr_II = error * ratio_II

        
        series_dict['I'] = part_I + corr_I
        series_dict['III'] = part_III + corr_III
        
        series_dict['II'][:n] = part_II - corr_II

    return series_dict


def fix_augmented_leads_internal(series_dict, alpha=0.33):
    if all(k in series_dict for k in ['aVR', 'aVL', 'aVF']):
        avr = series_dict['aVR']
        avl = series_dict['aVL']
        avf = series_dict['aVF']
        current_sum = avr + avl + avf
        delta = -1 * current_sum * alpha 
        series_dict['aVR'] = avr + delta
        series_dict['aVL'] = avl + delta
        series_dict['aVF'] = avf + delta
        
    return series_dict




def series_to_kaggle_format(series, sample_id, fs, sig_len, average_lead_ii_mode=None):


    rows = []

    lead_layout = [
        ['I', 'aVR', 'V1', 'V4'],
        ['II', 'aVL', 'V2', 'V5'],
        ['III', 'aVF', 'V3', 'V6'],
    ]

    points_per_2_5s = int(2.5 * fs)
    series_by_lead = {}

    for row_idx in range(3):
        row_signal = series[row_idx]
        leads = lead_layout[row_idx]
        ##########
        # if row_idx == 1:
        #     lengths = [len(row_signal) - 3 * points_per_2_5s,
        #               points_per_2_5s, points_per_2_5s, points_per_2_5s]
        # else:
        #     lengths = [points_per_2_5s] * 4

        # indices = np.cumsum(lengths)[:-1]
        # segments = np.split(row_signal, indices)

        # for lead, segment in zip(leads, segments):
        #     series_by_lead[lead] = segment

        segments = np.array_split(row_signal, 4)

        for lead, segment in zip(leads, segments):
            series_by_lead[lead] = segment


    ####################
    if 1:
        series_by_lead = dw(series_by_lead, alpha=0.33)
    ####################

    if average_lead_ii_mode == 'average':
        short_ii = series_by_lead['II']
        long_ii = series[3]
        overlap_len = min(len(short_ii), len(long_ii))
        averaged_head = (short_ii[:overlap_len] + long_ii[:overlap_len]) / 2.0
        final_ii = np.concatenate([averaged_head, long_ii[overlap_len:]])
        series_by_lead['II'] = final_ii
    else:
        series_by_lead['II'] = series[3]


    
    for lead in ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
        if lead not in series_by_lead:
            continue

        signal = series_by_lead[lead]
        for row_idx, value in enumerate(signal):
            rows.append({
                'id': f'{sample_id}_{row_idx}_{lead}',
                'fs': fs,
                'value': float(value)
            })

    return rows


def mask_to_prediction(mask, sample_id, fs, sig_len, cfg):
    t0, t1 = cfg.time_range
    cropped_pixel = mask[..., t0:t1]

    series_in_pixel = pixel_to_series(cropped_pixel, cfg.zero_mv_positions, sig_len)
    series = (np.array(cfg.zero_mv_positions).reshape(4, 1) - series_in_pixel) / cfg.mv_to_pixel

    pred_rows = series_to_kaggle_format(series, sample_id, fs, sig_len)

    return pred_rows

