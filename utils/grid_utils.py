import torch
import torch.nn.functional as F
import cv2
import numpy as np
import cc3d
from scipy.interpolate import griddata

def ROUND(x):
    if isinstance(x, list):
        return [int(round(xx)) for xx in x]
    else:
        return int(round(x))

def rectify_image(image, gridpoint_xy, target_height=1700, target_width=2200):
    H, W = target_height, target_width

    H1, W1 = image.shape[:2]
    sparse_map = gridpoint_xy / [[[W1 - 1, H1 - 1]]] * 2 - 1
    sparse_map = torch.from_numpy(np.ascontiguousarray(sparse_map.transpose(2, 0, 1))).unsqueeze(0).float()
    dense_map = F.interpolate(sparse_map, size=(H, W), mode='bilinear', align_corners=True)
    distort = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).float()
    rectified = F.grid_sample(
        distort, dense_map.permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False
    )
    rectified = rectified.data.cpu().numpy()
    rectified = rectified[0].transpose(1, 2, 0).astype(np.uint8)
    return rectified

def interpolate_mapping(gridpoint_xy):
    mx, my = np.meshgrid(np.arange(0, 57), np.arange(0, 44))
    coord = np.stack([mx, my], axis=-1).reshape(-1, 2)
    value = gridpoint_xy.copy()
    value = value.reshape(-1, 2)
    missing = np.all(value == [0, 0], axis=1)
    interpolate_xy = griddata(coord[~missing], value[~missing], (mx, my), method='cubic')
    interpolate_xy[np.isnan(interpolate_xy)] = 0
    return interpolate_xy

def segment_to_endpoints_fitline(mask):
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
    theta = np.arctan2(y2 - y1, x2 - x1)
    nx, ny = -np.sin(theta), np.cos(theta)
    rho = x1 * nx + y1 * ny
    return theta, rho, (nx, ny)

def compare_segment(seg1, seg2, angle_thr=np.deg2rad(3), rho_thr=10, gap_thr=15):
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
    if (y2 < y1):
        x1, y1, x2, y2 = x2, y2, x1, y1
    return x1, y1, x2, y2

def canonical_x_order(x1, y1, x2, y2):
    if (x2 < x1):
        x1, y1, x2, y2 = x2, y2, x1, y1
    return x1, y1, x2, y2

def output_to_predict(image, batch, output):
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

    cc = cc3d.connected_components(gridpoint > 0.5)
    stats = cc3d.statistics(cc)
    point_yx = stats['centroids'][1:]
    point_yx = np.round(point_yx).astype(np.int32)

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
            assert (len(area) == len(label))

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
            assert (len(area) == len(label))

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
    more = {
        'ghfiltered': ghfiltered,
        'gvfiltered': gvfiltered,
    }
    return gridpoint_xy, more
