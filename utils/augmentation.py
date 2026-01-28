import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

from noise import pnoise2

class AddPerlinDirt(ImageOnlyTransform):
    def __init__(self, scale_factor=4, p=0.5):
        super().__init__(p=p)
        self.scale_factor = scale_factor

    def apply(self, img, **params):
        return self._add_dirt(img)

    def _fast_noise(self, H, W):
        small_H, small_W = H // self.scale_factor, W // self.scale_factor
        if small_H == 0 or small_W == 0: 
            return np.zeros((H, W), dtype=np.float32)
        
        noise = np.random.rand(small_H, small_W).astype(np.float32)
        noise = cv2.resize(noise, (W, H), interpolation=cv2.INTER_LINEAR)
        
        k = max(int(self.scale_factor), 3)
        if k % 2 == 0: k += 1
        noise = cv2.GaussianBlur(noise, (k, k), 0)
        
        mn, mx = noise.min(), noise.max()
        if mx - mn > 1e-5: 
            noise = (noise - mn) / (mx - mn)
        else: 
            noise = np.zeros_like(noise)
        return noise
    
    # def _perlin_noise(self, H, W):
    #     small_H, small_W = H // self.scale_factor, W // self.scale_factor
    #     if small_H == 0 or small_W == 0:
    #         return np.zeros((H, W), dtype=np.float32)

    #     scale = np.random.uniform(10, 20)
    #     octaves = np.random.randint(2, 5)
    #     persistence = np.random.uniform(0.3, 0.6)
    #     lacunarity = np.random.uniform(1.5, 2.5)
    #     seed = np.random.randint(0, 1000)

    #     arr = np.zeros((small_H, small_W), dtype=np.float32)
    #     for i in range(small_H):
    #         for j in range(small_W):
    #             arr[i, j] = pnoise2(
    #                 i / scale, j / scale,
    #                 octaves=octaves,
    #                 persistence=persistence,
    #                 lacunarity=lacunarity,
    #                 base=seed,
    #                 repeatx=1024, repeaty=1024
    #             )

    #     if arr.max() - arr.min() != 0:
    #         arr = (arr - arr.min()) / (arr.max() - arr.min())

    #     return cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)

    # def _add_dirt(self, img):
    #     H, W, C = img.shape
    #     img_float = img.astype(np.float32)

    #     noise = self._perlin_noise(H, W)

    #     power_val = np.random.uniform(1.5, 2.5)

    #     noise = np.clip(noise, 0, 1)
    #     noise = np.power(noise, power_val)

    #     thresh = np.random.uniform(0.3, 0.8)
    #     mask_dirt = (noise > thresh).astype(np.float32) * noise

    #     k = np.random.randint(7, 50) | 1
    #     mask_blur = cv2.GaussianBlur(mask_dirt, (k, k), 0)

    #     if mask_blur.max() > 0:
    #         mask_blur = mask_blur / mask_blur.max()

    #     if np.random.rand() < 0.7:
    #         color = np.array([
    #             np.random.uniform(0.6, 0.8),
    #             np.random.uniform(0.5, 0.7),
    #             np.random.uniform(0.4, 0.6)
    #         ], dtype=np.float32)
    #     else:
    #         color = np.array([
    #             np.random.uniform(0.3, 0.5),
    #             np.random.uniform(0.4, 0.6),
    #             np.random.uniform(0.3, 0.5)
    #         ], dtype=np.float32)

    #     opacity = np.random.uniform(0.3, 0.8)
    #     stain_map = 1 - (mask_blur[..., None] * opacity * (1 - color))

    #     res = img_float * stain_map
    #     return np.clip(res, 0, 255).astype(np.uint8)
    def _add_dirt(self, img):
        H, W, C = img.shape
        img_float = img.astype(np.float32)
        noise = self._fast_noise(H, W)
        noise = np.clip(noise, 0, 1)
        power_val = np.random.uniform(1.5, 2.5)
        noise = np.power(noise, power_val)

        thresh = np.random.uniform(0.3, 0.8)
        mask_dirt = (noise > thresh).astype(np.float32) * noise

        k = np.random.randint(7, 50) | 1
        mask_blur = cv2.GaussianBlur(mask_dirt, (k, k), 0)

        if mask_blur.max() > 0:
            mask_blur = mask_blur / mask_blur.max()

        if np.random.rand() < 0.7:
            color = np.array([
                np.random.uniform(0.6, 0.8),
                np.random.uniform(0.5, 0.7),
                np.random.uniform(0.4, 0.6)
            ], dtype=np.float32)
        else:
            color = np.array([
                np.random.uniform(0.3, 0.5),
                np.random.uniform(0.4, 0.6),
                np.random.uniform(0.3, 0.5)
            ], dtype=np.float32)

        opacity = np.random.uniform(0.3, 0.8)
        stain_map = 1 - (mask_blur[..., None] * opacity * (1 - color))

        res = img_float * stain_map
        return np.clip(res, 0, 255).astype(np.uint8)

def get_train_augmentation(cfg):
    """
    Get training augmentation pipeline.
    All augmentations are enabled by default.
    """
    transforms = []

    transforms.append(
        A.OneOf([
            A.GridDropout(
                ratio=0.3,
                # unit_size_min=10,
                # unit_size_max=60,
                # holes_number_x=None,
                # holes_number_y=None,
                # fill_value=255,
                random_offset=True,
                p=1.0 
            ),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.01, 0.1),
                hole_width_range=(0.01, 0.1),
                fill=255,
                p=1.0
            ),
        ], p=0.2)
    )

    # Perlin Dirt: Adds realistic stain/dirt patterns using Perlin noise
    transforms.append(AddPerlinDirt(scale_factor=32, p=0.3))

    # Quality degradation: Simulates low-quality scans, blur, compression, or noise
    transforms.append(
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.Downscale(scale_range=(0.6, 0.85), p=1.0),  # Reduce resolution then upscale
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),  # Blur effect
            A.ImageCompression(quality_range=(30, 70), p=1.0),  # JPEG compression artifacts
            A.GaussNoise(std_range=(0.04, 0.15), per_channel=False, p=1.0),  # Gaussian noise
            A.ISONoise(color_shift=(0.1, 0.2), intensity=(0.5, 0.8), p=1.0),  # Camera sensor noise
        ], p=0.5)
    )

    # Color/appearance variations: Grayscale, brightness, contrast, hue, saturation, shadow
    transforms.append(
        A.OneOf([
            A.ToGray(p=1.0),  # Convert to grayscale
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0
            ),  # Adjust brightness and contrast
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=15,
                p=1.0
            ),  # Adjust hue, saturation, value
            A.RandomShadow(
                num_shadows_limit=(1, 2),
                shadow_dimension=4,
                shadow_roi=(0, 0, 1, 1),
                p=1.0
            ),  # Add random shadows
        ], p=0.5)
    )

    # Elastic Transform: Non-linear geometric distortion (simulates paper warping)
    transforms.append(
        A.ElasticTransform(
            alpha=20,
            sigma=65,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
            fill_mask=0,
            p=0.1
        )
    )

    transforms.append(A.HorizontalFlip(p=0.5))
    transforms.append(A.VerticalFlip(p=0.5))
    
    if not transforms:
        return None

    return A.Compose(transforms, p=1.0)


def get_val_augmentation(cfg):
    return None
