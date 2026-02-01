#!/usr/bin/env python3
"""
Generate Stage 2 masks (4352x1696) from CSV signal data.

This script creates 4-channel pixel masks from ECG signal CSV files.
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import argparse

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Get project root directory (parent of data_preparation)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class CFG:
    # Data paths
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    TRAIN_DIR = os.path.join(PROJECT_ROOT, 'data', 'train')
    CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'train.csv')
    TRAIN_FOLDS_PATH = os.path.join(PROJECT_ROOT, 'data', 'train_folds.csv')
    
    # Output directory
    MASK_DIR = os.path.join(PROJECT_ROOT, 'data', 'mask_4352x1696')
    
    # Mask dimensions (from stage2 config)
    CROP_X_RANGE = (0, 4352)
    CROP_Y_RANGE = (0, 1696)
    TIME_RANGE = (236, 4160)
    ZERO_MV_POSITIONS = [703.5, 987.5, 1271.5, 1531.5]
    MV_TO_PIXEL = 79
    
    # Processing
    NUM_WORKERS = 16


# ==============================================================================
# LEAD LAYOUT
# ==============================================================================

# Lead layout for mask generation
LAYOUT_3X4 = [
    (0, 0, 'I', 0),
    (0, 1, 'aVR', 0),
    (0, 2, 'V1', 0),
    (0, 3, 'V4', 0),
    (1, 0, 'II', 1),
    (1, 1, 'aVL', 1),
    (1, 2, 'V2', 1),
    (1, 3, 'V5', 1),
    (2, 0, 'III', 2),
    (2, 1, 'aVF', 2),
    (2, 2, 'V3', 2),
    (2, 3, 'V6', 2),
    (3, -1, 'II', 3),
]


# ==============================================================================
# MASK GENERATION
# ==============================================================================

def generate_stage2_mask(csv_path, fs, sig_len, output_size=(1696, 4352)):
    """
    Generate 4-channel pixel mask from CSV signal data.
    
    Args:
        csv_path: Path to CSV file with signal data
        fs: Sampling frequency
        sig_len: Signal length
        output_size: (H, W) = (1696, 4352)
    
    Returns:
        mask: (4, H, W) float32 array
    """
    H, W = output_size
    THICKNESS = 1
    
    mask_highres = np.zeros((4, H, sig_len), dtype=np.uint8)

    df = pd.read_csv(csv_path)

    if len(df) != sig_len:
        print(f"Warning: CSV has {len(df)} rows but sig_len={sig_len}")

    points_per_2_5s = int(2.5 * fs)

    for ch_idx, seg_idx, lead_name, row_idx in LAYOUT_3X4:
        if lead_name not in df.columns:
            print(f"Warning: {lead_name} not in CSV columns")
            continue

        if seg_idx == -1:
            s_start, s_end = 0, len(df)
        else:
            s_start = seg_idx * points_per_2_5s
            s_end = s_start + points_per_2_5s

            if s_start >= len(df):
                continue
            s_end = min(s_end, len(df))

        signal = df[lead_name].values[s_start:s_end]

        valid_mask = ~np.isnan(signal)
        if not valid_mask.any():
            continue

        valid_indices = np.where(valid_mask)[0]
        signal_valid = signal[valid_mask]

        x_coords = (s_start + valid_indices).astype(np.float32)

        y_coords = CFG.ZERO_MV_POSITIONS[ch_idx] - (signal_valid * CFG.MV_TO_PIXEL)
        y_coords = np.clip(y_coords, 0, H - 1)

        points = np.column_stack([x_coords, y_coords]).astype(np.int32)

        cv2.polylines(
            mask_highres[ch_idx],
            [points],
            isClosed=False,
            color=255,
            thickness=THICKNESS,
            lineType=cv2.LINE_AA,
        )

    # Resize to time range
    TOTAL_TIME_WIDTH = CFG.TIME_RANGE[1] - CFG.TIME_RANGE[0]
    mask_time = np.zeros((4, H, TOTAL_TIME_WIDTH), dtype=np.uint8)
    for ch_idx in range(4):
        mask_time[ch_idx] = cv2.resize(
            mask_highres[ch_idx],
            (TOTAL_TIME_WIDTH, H),
            interpolation=cv2.INTER_LINEAR
        )

    # Place in full width
    mask = np.zeros((4, H, W), dtype=np.uint8)
    mask[:, :, CFG.TIME_RANGE[0]:CFG.TIME_RANGE[1]] = mask_time

    mask = mask.astype(np.float32) / 255.0

    return mask


# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def process_single_sample(sample_row, train_dir, mask_dir, csv_path_global):
    """
    Process a single sample to generate mask.
    
    Returns:
        success: bool
        sample_id: str
        error_msg: str or None
    """
    sample_id = str(sample_row['id'])
    
    sample_dir = os.path.join(train_dir, sample_id)
    if not os.path.exists(sample_dir):
        return False, sample_id, f"Sample dir not found: {sample_dir}"
    
    # Read CSV for mask generation
    csv_path = os.path.join(sample_dir, f'{sample_id}.csv')
    if not os.path.exists(csv_path):
        return False, sample_id, f"CSV not found: {csv_path}"
    
    # Get metadata
    try:
        train_df = pd.read_csv(csv_path_global)
        meta = train_df[train_df['id'] == int(sample_id)].iloc[0]
        fs = int(meta['fs'])
        sig_len = int(meta['sig_len'])
    except Exception as e:
        return False, sample_id, f"Error reading metadata: {e}"
    
    # Generate mask
    try:
        mask_path = os.path.join(mask_dir, f'{sample_id}.mask.npy')
        if os.path.exists(mask_path):
            return True, sample_id, None  # Already exists
        
        mask = generate_stage2_mask(
            csv_path, fs, sig_len, 
            output_size=(CFG.CROP_Y_RANGE[1], CFG.CROP_X_RANGE[1])
        )
        np.save(mask_path, mask.astype(np.float32))
        return True, sample_id, None
        
    except Exception as e:
        return False, sample_id, f"Error generating mask: {e}"


def main(num_workers=16, debug=False, debug_samples=10):
    """Main processing function."""
    # Create output directory
    os.makedirs(CFG.MASK_DIR, exist_ok=True)
    
    # Load sample list
    train_folds_df = pd.read_csv(CFG.TRAIN_FOLDS_PATH)
    
    if debug:
        train_folds_df = train_folds_df.head(debug_samples)
        print(f"Debug mode: Processing only {len(train_folds_df)} samples")
    
    print(f"Processing {len(train_folds_df)} samples with {num_workers} workers...")
    print(f"Output: {CFG.MASK_DIR}")
    print(f"Mask size: {CFG.CROP_Y_RANGE[1]} x {CFG.CROP_X_RANGE[1]}")
    
    # Process samples
    success_count = 0
    failed_samples = []
    
    if num_workers > 1:
        process_func = partial(
            process_single_sample,
            train_dir=CFG.TRAIN_DIR,
            mask_dir=CFG.MASK_DIR,
            csv_path_global=CFG.CSV_PATH
        )
        
        with Pool(processes=num_workers) as pool:
            for success, sample_id, error_msg in tqdm(
                pool.imap(process_func, [row for _, row in train_folds_df.iterrows()]),
                total=len(train_folds_df),
                desc='Generating masks'
            ):
                if success:
                    success_count += 1
                else:
                    failed_samples.append((sample_id, error_msg))
    else:
        for _, row in tqdm(train_folds_df.iterrows(), total=len(train_folds_df), desc='Generating masks'):
            success, sample_id, error_msg = process_single_sample(
                row, CFG.TRAIN_DIR, CFG.MASK_DIR, CFG.CSV_PATH
            )
            if success:
                success_count += 1
            else:
                failed_samples.append((sample_id, error_msg))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(train_folds_df)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(failed_samples)}")
    
    if failed_samples:
        print(f"\nFailed samples:")
        for sample_id, error_msg in failed_samples[:10]:
            print(f"  {sample_id}: {error_msg}")
        if len(failed_samples) > 10:
            print(f"  ... and {len(failed_samples) - 10} more")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Stage 2 masks from CSV signals')
    parser.add_argument('--workers', type=int, default=16, help='Number of worker processes')
    parser.add_argument('--debug', action='store_true', help='Debug mode (process only first 10 samples)')
    parser.add_argument('--debug-samples', type=int, default=10, help='Number of samples in debug mode')
    
    args = parser.parse_args()
    
    main(
        num_workers=args.workers,
        debug=args.debug,
        debug_samples=args.debug_samples
    )