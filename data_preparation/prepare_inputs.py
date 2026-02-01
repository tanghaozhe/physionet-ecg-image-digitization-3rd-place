# https://www.kaggle.com/code/hengck23/demo-submission

#!/usr/bin/env python3
"""
Prepare Stage 2 inputs: rectified images (4400x1700).

This script:
1. Stage 0: Keypoint detection + normalization
2. Stage 1: Grid detection + perspective rectification
3. High-resolution rectification (natural size -> 4400x1700)

All model definitions and helper functions are self-contained.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.resnet import *
from scipy.interpolate import griddata
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import cc3d
import gc

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Get project root directory (parent of data_preparation)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class CFG:
    # Data paths
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    TRAIN_DIR = os.path.join(PROJECT_ROOT, 'data', 'train')
    TRAIN_FOLDS_PATH = os.path.join(PROJECT_ROOT, 'data', 'train_folds.csv')
    
    # Checkpoint paths
    STAGE0_CHECKPOINT = os.path.join(PROJECT_ROOT, 'checkpoints', 'stage0-last.checkpoint.pth')
    STAGE1_CHECKPOINT = os.path.join(PROJECT_ROOT, 'checkpoints', 'stage1-last.checkpoint.pth')
    
    # Output directories
    STAGE0_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'stage0', 'normalized_kaggle_data')
    STAGE1_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'stage1', 'rectified_kaggle_data_grid')
    RECTIFIED_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'stage1', 'rectified_kaggle_data_4400x1700')
    
    # Image dimensions
    BASE_WIDTH = 1440
    BASE_HEIGHT = 1152
    BASE_RECT_W = 2200
    BASE_RECT_H = 1700
    FIXED_RECT_WIDTH = 4400
    FIXED_RECT_HEIGHT = 1700
    
    # Processing
    DEVICE = 'cuda'
    FLOAT_TYPE = torch.float16
    NUM_WORKERS = 8


# ==============================================================================
# GLOBAL CONSTANTS
# ==============================================================================

LEAD_NAME_TO_LABEL = {
    'None': 0,
    'I': 1, 'aVR': 2, 'V1': 3, 'V4': 4,
    'II': 5, 'aVL': 6, 'V2': 7, 'V5': 8,
    'III': 9, 'aVF': 10, 'V3': 11, 'V6': 12,
    'II-rhythm': 13,
}
LABEL_TO_LEAD_NAME = {v: k for k, v in LEAD_NAME_TO_LABEL.items()}


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def ROUND(x):
    """Round to nearest integer."""
    if isinstance(x, list):
        return [int(round(xx)) for xx in x]
    else:
        return int(round(x))


def load_net(net, checkpoint_file):
    """Load model checkpoint."""
    f = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
    state_dict = f['state_dict']
    print(net.load_state_dict(state_dict, strict=False))
    net.eval()
    net.output_type = ['infer']
    return net


# ==============================================================================
# STAGE 0 MODEL
# ==============================================================================

class Stage0DecoderBlock(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel, scale=2):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class Stage0UnetDecoder(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel, scale=[2, 2, 2, 2]):
        super().__init__()
        self.center = nn.Identity()
        i_channel = [in_channel] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            Stage0DecoderBlock(i, s, o, sc)
            for i, s, o, sc in zip(i_channel, s_channel, o_channel, scale)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s)
            decode.append(d)
        last = d
        return last, decode


def encode_with_resnet_stage0(e, x):
    encode = []
    x = e.conv1(x)
    x = e.bn1(x)
    x = e.act1(x)
    x = e.layer1(x); encode.append(x)
    x = e.layer2(x); encode.append(x)
    x = e.layer3(x); encode.append(x)
    x = e.layer4(x); encode.append(x)
    return encode


class Stage0Net(nn.Module):
    def __init__(self, pretrained=True, cfg=None):
        super(Stage0Net, self).__init__()
        self.output_type = ['infer', 'loss']
        self.register_buffer('D', torch.tensor(0))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        arch = 'resnet18d.ra4_e3600_r224_in1k'
        encoder_dim = [64, 128, 256, 512]
        decoder_dim = [256, 128, 64, 32]

        self.encoder = timm.create_model(
            model_name=arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool=''
        )
        self.decoder = Stage0UnetDecoder(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1] + [0],
            out_channel=decoder_dim,
            scale=[2, 2, 2, 2]
        )
        self.marker = nn.Conv2d(decoder_dim[-1], 13 + 1, kernel_size=1)
        self.orientation = nn.Linear(encoder_dim[-1], 8)

    def forward(self, batch):
        device = self.D.device
        image = batch['image'].to(device)
        B, _3_, H, W = image.shape
        x = image.float() / 255
        x = (x - self.mean) / self.std

        e = self.encoder
        encode = encode_with_resnet_stage0(e, x)
        pooled = F.adaptive_avg_pool2d(encode[-1], 1).reshape(B, -1)

        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1] + [None]
        )

        marker = self.marker(last)
        orientation = self.orientation(pooled)

        output = {}
        if 'loss' in self.output_type:
            output['marker_loss'] = F.cross_entropy(marker, batch['marker'].to(device))
            output['orientation_loss'] = F.cross_entropy(orientation, batch['orientation'].to(device))

        if 'infer' in self.output_type:
            output['marker'] = torch.softmax(marker, 1)
            output['orientation'] = torch.softmax(orientation, 1)

        return output


# ==============================================================================
# STAGE 1 MODEL
# ==============================================================================

class Stage1DecoderBlock(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel, scale=2):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class Stage1UnetDecoder(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel, scale=[2, 2, 2, 2]):
        super().__init__()
        self.center = nn.Identity()
        i_channel = [in_channel] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            Stage1DecoderBlock(i, s, o, sc)
            for i, s, o, sc in zip(i_channel, s_channel, o_channel, scale)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s)
            decode.append(d)
        last = d
        return last, decode


def encode_with_resnet_stage1(e, x):
    encode = []
    x = e.conv1(x)
    x = e.bn1(x)
    x = e.act1(x)
    x = e.layer1(x); encode.append(x)
    x = e.layer2(x); encode.append(x)
    x = e.layer3(x); encode.append(x)
    x = e.layer4(x); encode.append(x)
    return encode


class Stage1Net(nn.Module):
    def __init__(self, pretrained=True, cfg=None):
        super(Stage1Net, self).__init__()
        self.output_type = ['infer', 'loss']
        self.register_buffer('D', torch.tensor(0))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        arch = 'resnet34.a3_in1k'
        encoder_dim = [64, 128, 256, 512]
        decoder_dim = [256, 128, 64, 32]

        self.encoder = timm.create_model(
            model_name=arch, pretrained=pretrained, in_chans=3, num_classes=0, global_pool=''
        )
        self.decoder = Stage1UnetDecoder(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1] + [0],
            out_channel=decoder_dim,
            scale=[2, 2, 2, 2]
        )
        self.gridpoint = nn.Conv2d(decoder_dim[-1], 1, kernel_size=1)
        self.gridhline = nn.Conv2d(decoder_dim[-1], 44 + 1, kernel_size=1)
        self.gridvline = nn.Conv2d(decoder_dim[-1], 57 + 1, kernel_size=1)
        self.marker = nn.Conv2d(decoder_dim[-1], 13 + 1, kernel_size=1)

    def forward(self, batch):
        device = self.D.device
        image = batch['image'].to(device)
        B, _3_, H, W = image.shape
        x = image.float() / 255
        x = (x - self.mean) / self.std

        e = self.encoder
        encode = encode_with_resnet_stage1(e, x)

        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1] + [None]
        )

        marker = self.marker(last)
        gridpoint = self.gridpoint(last)
        gridhline = self.gridhline(last)
        gridvline = self.gridvline(last)

        output = {}
        if 'loss' in self.output_type:
            output['marker_loss'] = F.cross_entropy(marker, batch['marker'].to(device))
            output['gridpoint_loss'] = F.binary_cross_entropy_with_logits(
                gridpoint, batch['gridpoint'].to(device),
                pos_weight=torch.tensor([10]).to(device)
            )
            output['gridhline_loss'] = F.cross_entropy(gridhline, batch['gridhline'].to(device))
            output['gridvline_loss'] = F.cross_entropy(gridvline, batch['gridvline'].to(device))

        if 'infer' in self.output_type:
            output['marker'] = torch.softmax(marker, 1)
            output['gridpoint'] = torch.sigmoid(gridpoint)
            output['gridhline'] = torch.softmax(gridhline, 1)
            output['gridvline'] = torch.softmax(gridvline, 1)

        return output


# ==============================================================================
# STAGE 0 POST-PROCESSING
# ==============================================================================

def image_to_batch(image):
    """Prepare image batch with TTA."""
    H, W = image.shape[:2]
    scale = CFG.BASE_WIDTH / W

    simage = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sH, sW = simage.shape[:2]
    pH = int(sH // 32) * 32 + 32
    pW = int(sW // 32) * 32 + 32
    padded = np.pad(simage, [[0, pH - sH], [0, pW - sW], [0, 0]], mode='constant', constant_values=0)

    image0 = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).unsqueeze(0)
    batch = {'image': [image0], 'tta': [0]}

    for tta in [1, 2, 3]:
        if tta == 1:
            image1 = torch.flip(image0, [2]).contiguous()
        if tta == 2:
            image1 = torch.flip(image0, [3]).contiguous()
        if tta == 3:
            image1 = torch.flip(image0, [2, 3]).contiguous()
        batch['image'].append(image1)
        batch['tta'].append(tta)

    batch['image'] = torch.cat(batch['image'])
    batch['scale'] = scale
    batch['sW'] = sW
    batch['sH'] = sH
    batch['W'] = W
    batch['H'] = H
    return batch


def marker_to_keypoint(image, orientation, marker, scale):
    """Convert marker prediction to keypoints."""
    orientation = orientation.data.cpu().numpy().reshape(-1)
    marker = marker.permute(0, 2, 3, 1).float().data.cpu().numpy()[0]

    k = orientation.argmax()
    if k != 0:
        if k <= 3:
            k = -k
        else:
            print(f'k={k} rotation unknown????')

    marker = np.rot90(marker, k, axes=(0, 1))
    keypoint = []
    thresh = marker.argmax(-1)
    for label in [2, 3, 4, 6, 7, 8, 10, 11, 12]:
        cc = cc3d.connected_components(thresh == label)
        stats = cc3d.statistics(cc)
        center = stats['centroids'][1:]
        area = stats['voxel_counts'][1:]
        argsort = np.argsort(area)[::-1]
        center = center[argsort]
        area = area[argsort]

        center = np.append(center, [[0, 0]], axis=0)
        area = np.append(area, [1], axis=0)

        for (y, x), a in zip(center[:1], area[:1]):
            leadname = LABEL_TO_LEAD_NAME[label]
            x, y = x / scale, y / scale
            keypoint.append([x, y, label, leadname])

    return keypoint, k


def output_to_predict_stage0(image, batch, output):
    """Process Stage 0 output."""
    marker = 0
    orientation = 0

    num_tta = len(batch['tta'])
    sH, sW = batch['sH'], batch['sW']
    scale = batch['scale']

    for b in range(num_tta):
        tta = batch['tta'][b]
        mk = output['marker'][[b]]
        on = output['orientation'][b]

        if tta == 1:
            mk = torch.flip(mk, [2]).contiguous()
            on = on[[4, 5, 6, 7, 0, 1, 2, 3]]
        elif tta == 2:
            mk = torch.flip(mk, [3]).contiguous()
            on = on[[6, 7, 4, 5, 2, 3, 0, 1]]
        elif tta == 3:
            mk = torch.flip(mk, [2, 3]).contiguous()
            on = on[[2, 3, 0, 1, 6, 7, 4, 5]]
        else:
            pass

        orientation += on
        marker += mk[..., :sH, :sW]

    marker = marker / num_tta
    orientation = orientation / num_tta
    keypoint, k = marker_to_keypoint(image, orientation, marker, scale)
    rotated = np.ascontiguousarray(np.rot90(image, k, axes=(0, 1)))
    return rotated, keypoint, k


# Reference point for normalization
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GRIDPOINT_REF_PATH = os.path.join(THIS_DIR, '..', 'frog', '640106434-0001.gridpoint_xy.npy')

if os.path.exists(GRIDPOINT_REF_PATH):
    gridpoint0001_xy = np.load(GRIDPOINT_REF_PATH)
else:
    gridpoint0001_xy = np.zeros((44, 57, 2), dtype=np.float32)


def make_ref_point():
    """Create reference points for homography."""
    h0001, w0001 = 1700, 2200
    ref_pt = []
    for j, i in [[19, 3], [26, 3], [33, 3]]:
        x, y = gridpoint0001_xy[j, i + 13]
        ref_pt.append([x, y])
        x, y = gridpoint0001_xy[j, i + 25]
        ref_pt.append([x, y])
        x, y = gridpoint0001_xy[j, i + 38]
        ref_pt.append([x, y])

    ref_pt = np.array(ref_pt, np.float32)
    scale = 1280 / w0001
    ref_pt = ref_pt * [[scale, scale]]
    shift = (1440 - 1280) / 2
    ref_pt = ref_pt + [[shift, shift]] + [[-6, +10]]
    return ref_pt


REF_PT9 = make_ref_point()


def normalise_image(image, pt9, ref_pt9=REF_PT9):
    """Apply perspective normalization."""
    pt9 = np.array(pt9, np.float32)
    homo, match = cv2.findHomography(pt9, ref_pt9, method=cv2.RANSAC)
    aligned = cv2.warpPerspective(image, homo, (CFG.BASE_WIDTH, CFG.BASE_HEIGHT))
    match = match.reshape(-1)
    return aligned, homo, match


def normalise_by_homography(image, keypoint):
    """Normalize image using keypoints."""
    pt9 = [[k[0], k[1]] for k in keypoint]
    normalised, homo, match = normalise_image(image, pt9)
    for i in range(len(keypoint)):
        keypoint[i].append(match[i])
    return normalised, keypoint, homo


# ==============================================================================
# STAGE 1 POST-PROCESSING
# ==============================================================================

def interpolate_mapping(gridpoint_xy):
    """Interpolate missing grid points."""
    mx, my = np.meshgrid(np.arange(0, 57), np.arange(0, 44))
    coord = np.stack([mx, my], axis=-1).reshape(-1, 2)
    value = gridpoint_xy.copy()
    value = value.reshape(-1, 2)
    missing = np.all(value == [0, 0], axis=1)
    if not missing.all():
        interpolate_xy = griddata(coord[~missing], value[~missing], (mx, my), method='cubic')
        interpolate_xy[np.isnan(interpolate_xy)] = 0
        return interpolate_xy
    return gridpoint_xy


def segment_to_endpoints_fitline(mask):
    """Fit line to segment."""
    ys, xs = np.nonzero(mask)
    if xs.size < 2:
        return None

    pts = np.column_stack([xs, ys]).astype(np.float32)

    if pts.shape[0] > 20000:
        idx = np.random.choice(pts.shape[0], 20000, replace=False)
        pts = pts[idx]

    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    u = np.array([vx, vy], dtype=np.float32)
    u /= (np.linalg.norm(u) + 1e-12)

    p0 = np.array([x0, y0], dtype=np.float32)

    t = ((pts - p0) @ u)
    t_min, t_max = t.min(), t.max()

    pA = p0 + t_min * u
    pB = p0 + t_max * u

    return int(round(pA[0])), int(round(pA[1])), int(round(pB[0])), int(round(pB[1]))


def line_params(x1, y1, x2, y2):
    """Compute line parameters."""
    theta = np.arctan2(y2 - y1, x2 - x1)
    nx, ny = -np.sin(theta), np.cos(theta)
    rho = x1 * nx + y1 * ny
    return theta, rho, (nx, ny)


def compare_segment(seg1, seg2, angle_thr=np.deg2rad(3), rho_thr=10, gap_thr=15):
    """Compare two segments."""
    x11, y11, x12, y12 = seg1
    x21, y21, x22, y22 = seg2

    theta1, ρ1, n1 = line_params(x11, y11, x12, y12)
    theta2, ρ2, n2 = line_params(x21, y21, x22, y22)

    dtheta = np.abs((theta1 - theta2 + np.pi / 2) % np.pi - np.pi / 2)
    dρ = abs(ρ1 - ρ2)

    dxy = np.min([
        np.abs(x11 - x21) + np.abs(y11 - y21),
        np.abs(x11 - x22) + np.abs(y11 - y22),
        np.abs(x12 - x21) + np.abs(y12 - y21),
        np.abs(x12 - x22) + np.abs(y12 - y22),
    ])

    dtheta = dtheta / np.pi * 180
    return dtheta, dρ, dxy


def canonical_y_order(x1, y1, x2, y2):
    """Order by y coordinate."""
    if y2 < y1:
        x1, y1, x2, y2 = x2, y2, x1, y1
    return x1, y1, x2, y2


def canonical_x_order(x1, y1, x2, y2):
    """Order by x coordinate."""
    if x2 < x1:
        x1, y1, x2, y2 = x2, y2, x1, y1
    return x1, y1, x2, y2


def output_to_predict_stage1(image, batch, output):
    """Process Stage 1 output to get grid points."""
    marker = output['marker'][0]
    gridpoint = output['gridpoint'][0, 0]
    gridhline = output['gridhline'][0]
    gridvline = output['gridvline'][0]

    marker = marker.argmax(0).byte().data.cpu().numpy()
    gridpoint = gridpoint.float().data.cpu().numpy()
    gridhline = gridhline.float().data.cpu().numpy()
    gridvline = gridvline.float().data.cpu().numpy()

    gridline = np.stack([1 - gridhline[0], 1 - gridvline[0]])
    gridvline = gridvline.argmax(0).astype(np.uint8)
    gridhline = gridhline.argmax(0).astype(np.uint8)

    # Point detection
    cc = cc3d.connected_components(gridpoint > 0.5)
    stats = cc3d.statistics(cc)
    point_yx = stats['centroids'][1:]
    point_yx = np.round(point_yx).astype(np.int32)

    # Vertical line processing
    gvfiltered = np.zeros_like(gridvline)
    cc = cc3d.connected_components(gridvline != 0)
    num_line = cc.max()
    for l in range(1, num_line + 1):
        t = (cc == l)
        bincount = np.bincount(gridvline[t])
        c = bincount.argmax()
        gvfiltered[t] = c

    gvreject = np.zeros_like(gridvline)
    for l in range(1, num_line + 1):
        cc_local = cc3d.connected_components(gvfiltered == l)
        if cc_local.max() > 1:
            num = cc_local.max() + 1
            stats = cc3d.statistics(cc_local)
            area = stats['voxel_counts'][1:]
            label = np.arange(1, num)

            argsort = np.argsort(area)[::-1]
            area = area[argsort]
            label = label[argsort]

            if area[0] < 7:
                continue
            main_segment = segment_to_endpoints_fitline(cc_local == label[0])
            main_segment = canonical_y_order(*main_segment)

            for j in range(1, len(label)):
                if area[j] < 7:
                    continue

                segment = segment_to_endpoints_fitline(cc_local == label[j])
                segment = canonical_y_order(*segment)
                ang_dis, ori_dis, seg_dis = compare_segment(main_segment, segment)

                if ori_dis > 5:
                    gvreject[cc_local == label[j]] = 255
                else:
                    gvfiltered[cc_local == label[j]] = l

    vcc = gvfiltered.copy()

    # Horizontal line processing
    ghfiltered = np.zeros_like(gridhline)
    cc = cc3d.connected_components(gridhline != 0)
    num_line = cc.max()
    for l in range(1, num_line + 1):
        t = (cc == l)
        bincount = np.bincount(gridhline[t])
        c = bincount.argmax()
        ghfiltered[t] = c

    ghreject = np.zeros_like(gridhline)
    for l in range(1, num_line + 1):
        cc_local = cc3d.connected_components(ghfiltered == l)
        if cc_local.max() > 1:
            num = cc_local.max() + 1
            stats = cc3d.statistics(cc_local)
            area = stats['voxel_counts'][1:]
            label = np.arange(1, num)

            argsort = np.argsort(area)[::-1]
            area = area[argsort]
            label = label[argsort]

            if area[0] < 7:
                continue
            main_segment = segment_to_endpoints_fitline(cc_local == label[0])
            main_segment = canonical_x_order(*main_segment)

            for j in range(1, len(label)):
                if area[j] < 7:
                    continue

                segment = segment_to_endpoints_fitline(cc_local == label[j])
                segment = canonical_x_order(*segment)
                ang_dis, ori_dis, seg_dis = compare_segment(main_segment, segment)

                if ori_dis > 5:
                    ghreject[cc_local == label[j]] = 255
                else:
                    ghfiltered[cc_local == label[j]] = l

    hcc = ghfiltered.copy()

    # Mapping
    gridpoint_xy = np.zeros((44, 57, 2), np.float32)
    for y, x in point_yx:
        uy = ROUND(y)
        ux = ROUND(x)
        j = hcc[uy, ux]
        i = vcc[uy, ux]
        if (j == 0) | (i == 0):
            continue
        gridpoint_xy[j - 1, i - 1] = [x, y]

    gridpoint_xy = interpolate_mapping(gridpoint_xy)
    return gridpoint_xy


# ==============================================================================
# HIGH-RESOLUTION RECTIFICATION
# ==============================================================================

def get_highres_homography(homo_low, scale):
    """Scale homography matrix."""
    S = np.array([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ])
    return S @ homo_low


def apply_rotation(image, k):
    """Apply rotation to image."""
    if k == 0:
        return image
    return np.ascontiguousarray(np.rot90(image, k, axes=(0, 1)))


def rectify_highres(normalized_img, grid_small, scale, target_shape):
    """Rectify image using grid."""
    target_h, target_w = target_shape
    H_norm, W_norm = normalized_img.shape[:2]

    grid_large = grid_small * scale

    grid_norm = grid_large / np.array([[[W_norm - 1, H_norm - 1]]]) * 2 - 1

    sparse_map = torch.from_numpy(
        np.ascontiguousarray(grid_norm.transpose(2, 0, 1))
    ).unsqueeze(0).float()

    dense_map = F.interpolate(
        sparse_map,
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=True
    )

    distort = torch.from_numpy(
        np.ascontiguousarray(normalized_img.transpose(2, 0, 1))
    ).unsqueeze(0).float()

    rectified = F.grid_sample(
        distort,
        dense_map.permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    )

    rectified = rectified[0].permute(1, 2, 0).byte().cpu().numpy()

    return rectified


def compute_fixed_size_rectified_img(ori_img_path, homo_path, gridpoint_path, rotation_path):
    """
    Compute high-resolution rectified image.
    
    Returns:
        rectified_natural: uint8 numpy array (H, W, 3) or None
        scale: float or None
    """
    if not (os.path.exists(ori_img_path) and os.path.exists(homo_path) and
            os.path.exists(gridpoint_path) and os.path.exists(rotation_path)):
        return None, None

    try:
        original_img = cv2.imread(ori_img_path)
        if original_img is None:
            return None, None

        k = int(np.load(rotation_path)[0])
        rotated_img = apply_rotation(original_img, k)

        H_rotated, W_rotated = rotated_img.shape[:2]
        scale = W_rotated / CFG.BASE_WIDTH

        homo = np.load(homo_path)
        homo_highres = get_highres_homography(homo, scale)

        norm_w = int(CFG.BASE_WIDTH * scale)
        norm_h = int(CFG.BASE_HEIGHT * scale)

        normalized_highres = cv2.warpPerspective(
            rotated_img,
            homo_highres,
            (norm_w, norm_h),
            flags=cv2.INTER_CUBIC
        )

        grid_small = np.load(gridpoint_path)

        target_rect_w_natural = int(CFG.BASE_RECT_W * scale)
        target_rect_h_natural = int(CFG.BASE_RECT_H * scale)

        rectified_natural = rectify_highres(
            normalized_highres,
            grid_small,
            scale,
            (target_rect_h_natural, target_rect_w_natural)
        )

        return rectified_natural, float(scale)

    except Exception as e:
        print(f"Error in compute_fixed_size_rectified_img: {e}")
        return None, None


# ==============================================================================
# WORKER INITIALIZATION (加载一次模型)
# ==============================================================================

# 全局变量，在每个 worker 进程中初始化
_worker_stage0_net = None
_worker_stage1_net = None
_worker_device = None

def init_worker():
    """在每个 worker 进程启动时加载模型（只执行一次）"""
    global _worker_stage0_net, _worker_stage1_net, _worker_device
    
    _worker_device = torch.device(CFG.DEVICE)
    
    # Load Stage 0 model
    _worker_stage0_net = Stage0Net(pretrained=False)
    _worker_stage0_net = load_net(_worker_stage0_net, CFG.STAGE0_CHECKPOINT)
    _worker_stage0_net.to(_worker_device)
    
    # Load Stage 1 model
    _worker_stage1_net = Stage1Net(pretrained=False)
    _worker_stage1_net = load_net(_worker_stage1_net, CFG.STAGE1_CHECKPOINT)
    _worker_stage1_net.to(_worker_device)
    
    print(f"[Worker {os.getpid()}] Models loaded successfully")


def process_single_image(image_file, sample_dir, sample_id, stage0_dir, stage1_dir, rectified_dir):
    """处理单张图片（使用已加载的模型）"""
    global _worker_stage0_net, _worker_stage1_net, _worker_device
    
    image_id = image_file.replace('.png', '')
    ori_img_path = os.path.join(sample_dir, image_file)
    
    # Define output paths
    norm_path = os.path.join(stage0_dir, f'{image_id}.norm.png')
    homo_path = os.path.join(stage0_dir, f'{image_id}.homo.npy')
    rotation_path = os.path.join(stage0_dir, f'{image_id}.rotation.npy')
    gridpoint_path = os.path.join(stage1_dir, f'{image_id}.gridpoint_xy.npy')
    rect_path = os.path.join(rectified_dir, f'{image_id}.rect.png')
    
    # Skip if already processed
    if os.path.exists(rect_path):
        return True, image_id, "Already exists"
    
    try:
        # Read original image
        image = cv2.imread(ori_img_path, cv2.IMREAD_COLOR)
        if image is None:
            return False, image_id, "Failed to read image"
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Stage 0: Keypoint detection + Normalization
        batch = image_to_batch(image)
        
        with torch.amp.autocast('cuda', dtype=CFG.FLOAT_TYPE):
            with torch.no_grad():
                output = _worker_stage0_net(batch)
                rotated, keypoint, k = output_to_predict_stage0(image, batch, output)
                normalised, keypoint, homo = normalise_by_homography(rotated, keypoint)
        
        # Save Stage 0 outputs
        cv2.imwrite(norm_path, cv2.cvtColor(normalised, cv2.COLOR_RGB2BGR))
        np.save(homo_path, homo)
        np.save(rotation_path, np.array([k]))
        
        # Stage 1: Grid detection
        batch_s1 = {'image': torch.from_numpy(
            np.ascontiguousarray(normalised.transpose(2, 0, 1))
        ).unsqueeze(0)}
        
        with torch.amp.autocast('cuda', dtype=CFG.FLOAT_TYPE):
            with torch.no_grad():
                output = _worker_stage1_net(batch_s1)
                gridpoint_xy = output_to_predict_stage1(normalised, batch_s1, output)
        
        np.save(gridpoint_path, gridpoint_xy)
        
        # High-resolution rectification
        rectified_natural, scale = compute_fixed_size_rectified_img(
            ori_img_path, homo_path, gridpoint_path, rotation_path
        )
        
        if rectified_natural is None:
            return False, image_id, "Rectification failed"
        
        # Resize to fixed size (4400x1700)
        rectified_final = cv2.resize(
            rectified_natural,
            (CFG.FIXED_RECT_WIDTH, CFG.FIXED_RECT_HEIGHT),
            interpolation=cv2.INTER_CUBIC
        )
        
        cv2.imwrite(rect_path, rectified_final)
        
        return True, image_id, None
        
    except Exception as e:
        return False, image_id, str(e)


def process_single_sample(args):
    """
    Process a single sample through all stages.
    Args is a tuple: (sample_row, train_dir, stage0_dir, stage1_dir, rectified_dir)
    """
    sample_row, train_dir, stage0_dir, stage1_dir, rectified_dir = args
    
    sample_id = str(sample_row['id'])
    
    sample_dir = os.path.join(train_dir, sample_id)
    if not os.path.exists(sample_dir):
        return False, sample_id, f"Sample dir not found: {sample_dir}"
    
    # Get image files
    image_files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
    if len(image_files) == 0:
        return False, sample_id, "No PNG files found"
    
    # Process each image variant
    processed_count = 0
    for image_file in image_files:
        success, image_id, error_msg = process_single_image(
            image_file, sample_dir, sample_id, 
            stage0_dir, stage1_dir, rectified_dir
        )
        if success:
            processed_count += 1
    
    if processed_count > 0:
        return True, sample_id, None
    else:
        return False, sample_id, "No images processed"


def main(num_workers=8, debug=False, debug_samples=10):
    """Main processing function."""
    # Display project info
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Current working directory: {os.getcwd()}")
    print()
    
    # Create output directories
    os.makedirs(CFG.STAGE0_DIR, exist_ok=True)
    os.makedirs(CFG.STAGE1_DIR, exist_ok=True)
    os.makedirs(CFG.RECTIFIED_DIR, exist_ok=True)
    
    # Check if train_folds.csv exists
    if not os.path.exists(CFG.TRAIN_FOLDS_PATH):
        print(f"ERROR: {CFG.TRAIN_FOLDS_PATH} not found!")
        print(f"Please ensure data/train_folds.csv exists in the project root.")
        return
    
    # Load sample list
    train_folds_df = pd.read_csv(CFG.TRAIN_FOLDS_PATH)
    
    if debug:
        train_folds_df = train_folds_df.head(debug_samples)
        print(f"Debug mode: Processing only {len(train_folds_df)} samples")
    
    print(f"Processing {len(train_folds_df)} samples with {num_workers} workers...")
    print(f"Outputs (in project root):")
    print(f"  - Stage 0: {os.path.relpath(CFG.STAGE0_DIR, PROJECT_ROOT)}")
    print(f"  - Stage 1: {os.path.relpath(CFG.STAGE1_DIR, PROJECT_ROOT)}")
    print(f"  - Rectified: {os.path.relpath(CFG.RECTIFIED_DIR, PROJECT_ROOT)}")
    print()
    
    # Process samples
    success_count = 0
    failed_samples = []
    
    if num_workers > 1:
        # Prepare args for each sample
        sample_args = [
            (row, CFG.TRAIN_DIR, CFG.STAGE0_DIR, CFG.STAGE1_DIR, CFG.RECTIFIED_DIR)
            for _, row in train_folds_df.iterrows()
        ]
        
        with Pool(processes=num_workers, initializer=init_worker) as pool:
            for success, sample_id, error_msg in tqdm(
                pool.imap(process_single_sample, sample_args),
                total=len(train_folds_df),
                desc='Processing'
            ):
                if success:
                    success_count += 1
                else:
                    failed_samples.append((sample_id, error_msg))
    else:
        for _, row in tqdm(train_folds_df.iterrows(), total=len(train_folds_df), desc='Processing'):
            success, sample_id, error_msg = process_single_sample(
                row, CFG.TRAIN_DIR, CFG.STAGE0_DIR, 
                CFG.STAGE1_DIR, CFG.RECTIFIED_DIR
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare Stage 2 inputs (rectified images)')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker processes')
    parser.add_argument('--debug', action='store_true', help='Debug mode (process only first 10 samples)')
    parser.add_argument('--debug-samples', type=int, default=10, help='Number of samples in debug mode')
    
    args = parser.parse_args()
    
    main(
        num_workers=args.workers,
        debug=args.debug,
        debug_samples=args.debug_samples
    )