import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.augmentation import get_train_augmentation, get_val_augmentation


class Stage2Dataset(Dataset):
    def __init__(self, cfg, split='train', fold=None):
        self.cfg = cfg
        self.split = split

        if split == 'train' and cfg.use_augmentation:
            self.transform = get_train_augmentation(cfg)
        else:
            self.transform = get_val_augmentation(cfg)

        train_folds_path = os.path.join(cfg.data_dir, 'train_folds.csv')
        df = pd.read_csv(train_folds_path)

        if split == 'train':
            sample_ids = df[df['fold'] != fold]['id'].values
        else:
            sample_ids = df[df['fold'] == fold]['id'].values

        self.samples = []
        for idx, row in df[df['id'].isin(sample_ids)].iterrows():
            sample_id = str(row['id'])
            # if sample_id in ["2042290760"]:
            #     print(f"bad data sample_id:{sample_id}")
            #     continue

            fs = row['fs']
            sig_len = row['sig_len']

            mask_path = os.path.join(cfg.mask_dir, f'{sample_id}.mask.npy')
            if not os.path.exists(mask_path):
                continue

            sample_dir = os.path.join(cfg.data_dir, 'train', sample_id)
            if not os.path.exists(sample_dir):
                continue

            image_files = [f.replace('.png', '') for f in os.listdir(sample_dir) if f.endswith('.png')]

            for image_id in image_files:
                rect_path = os.path.join(cfg.rectified_dir, f'{image_id}.rect.png')
                if not os.path.exists(rect_path):
                    continue

                self.samples.append({
                    'sample_id': sample_id,
                    'image_id': image_id,
                    'rect_path': rect_path,
                    'mask_path': mask_path,
                    'fs': fs,
                    'sig_len': sig_len,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = cv2.imread(sample['rect_path'], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pixel_mask = np.load(sample['mask_path'])

        x0, x1 = self.cfg.crop_x_range
        y0, y1 = self.cfg.crop_y_range
        crop = image[y0:y1, x0:x1]

        if self.transform is not None:
            mask_hwc = pixel_mask.transpose(1, 2, 0)
            augmented = self.transform(image=crop, mask=mask_hwc)
            crop = augmented['image']
            mask_hwc = augmented['mask']
            pixel_mask = mask_hwc.transpose(2, 0, 1)

        batch = {
            'image': torch.from_numpy(crop.transpose(2, 0, 1)).byte(),
            'pixel': torch.from_numpy(pixel_mask).float(),
            'sample_id': sample['sample_id'],
            'fs': sample['fs'],
            'sig_len': sample['sig_len'],
        }

        return batch
