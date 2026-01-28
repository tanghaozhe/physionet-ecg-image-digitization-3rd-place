import torch
import cv2
import numpy as np
import cc3d

def ROUND(x):
    if isinstance(x, list):
        return [int(round(xx)) for xx in x]
    else:
        return int(round(x))

def load_checkpoint_for_inference(model, checkpoint_file, device='cuda'):
    f = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
    state_dict = f['state_dict']
    print(model.load_state_dict(state_dict, strict=False))

    model.to(device)
    model.eval()
    model.output_type = ['infer']
    return model

def load_stage0_checkpoint(model, checkpoint_file):
    return load_checkpoint_for_inference(model, checkpoint_file)

def make_ref_point(gridpoint0001_xy, h0001=1700, w0001=2200, target_width=1440):
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
    shift = (target_width - 1280) / 2
    ref_pt = ref_pt + [[shift, shift]] + [[-6, +10]]
    return ref_pt

def normalise_image(image, pt9, ref_pt9):
    pt9 = np.array(pt9, np.float32)
    homo, match = cv2.findHomography(pt9, ref_pt9, method=cv2.RANSAC)
    height, width = 1152, 1440
    aligned = cv2.warpPerspective(image, homo, (width, height))
    match = match.reshape(-1)
    return aligned, homo, match

def marker_to_keypoint(image, orientation, marker, scale, label_to_lead_name):
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
            leadname = label_to_lead_name[label]
            x, y = x / scale, y / scale
            keypoint.append([x, y, label, leadname])

    return keypoint, k

def image_to_batch(image, target_width=1440):
    H, W = image.shape[:2]
    scale = target_width / W

    simage = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sH, sW = simage.shape[:2]
    pH = int(sH // 32) * 32 + 32
    pW = int(sW // 32) * 32 + 32
    padded = np.pad(simage, [[0, pH - sH], [0, pW - sW], [0, 0]], mode='constant', constant_values=0)

    image0 = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).unsqueeze(0)
    batch = {
        'image': [image0],
        'tta': [0]
    }

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

def output_to_predict(image, batch, output, label_to_lead_name):
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
    keypoint, k = marker_to_keypoint(image, orientation, marker, scale, label_to_lead_name)
    rotated = np.ascontiguousarray(np.rot90(image, k, axes=(0, 1)))
    return rotated, keypoint, k

def normalise_by_homography(image, keypoint, ref_pt9):
    pt9 = [[k[0], k[1]] for k in keypoint]
    normalised, homo, match = normalise_image(image, pt9, ref_pt9)
    for i in range(len(keypoint)):
        keypoint[i].append(match[i])
    return normalised, keypoint, homo
