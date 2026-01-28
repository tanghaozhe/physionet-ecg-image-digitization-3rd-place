import cv2
import numpy as np

def show_image(image, name='image', mode=cv2.WINDOW_AUTOSIZE, resize=None):
    cv2.namedWindow(name, mode)
    if image.ndim == 3:
        cv2.imshow(name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if image.ndim == 2:
        cv2.imshow(name, image)
    if resize is not None:
        H, W = image.shape[:2]
        cv2.resizeWindow(name, int(resize * W), int(resize * H))

def draw_mapping(image, gridpoint_xy):
    overlay = image.copy() // 2
    for x, y in gridpoint_xy.reshape(-1, 2):
        y = int(round(y))
        x = int(round(x))
        if (x == 0) | (y == 0):
            continue
        cv2.circle(overlay, (x, y), 2, [0, 255, 0], -1)
    return overlay

def draw_lead_pixel(image, pixel):
    overlay = image // 2
    overlay = 255 - (255 - overlay) * (1 - pixel[0][..., np.newaxis] * [[[1, 0, 0]]])
    overlay = 255 - (255 - overlay) * (1 - pixel[1][..., np.newaxis] * [[[0, 1, 0]]])
    overlay = 255 - (255 - overlay) * (1 - pixel[2][..., np.newaxis] * [[[0, 0, 1]]])
    overlay = 255 - (255 - overlay) * (1 - pixel[3][..., np.newaxis] * [[[1, 1, 0]]])
    overlay = overlay.astype(np.uint8)
    return overlay
