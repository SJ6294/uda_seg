import random

import mmcv
import numpy as np

from ..builder import PIPELINES


def _ensure_2d_mask(seg):
    """Convert mask array to 2D if loaded as multi-channel.

    Some datasets store binary masks as RGB PNGs. In that case `seg` can be
    HxWx3 and cannot be copied into a 2D canvas directly.
    """
    if seg.ndim == 2:
        return seg
    if seg.ndim == 3:
        if seg.shape[2] == 1:
            return seg[:, :, 0]
        # Use the first channel for RGB-like masks.
        return seg[:, :, 0]
    raise ValueError(f'Unsupported mask ndim={seg.ndim}, shape={seg.shape}')


def _align_mask_to_image(seg, h, w):
    """Ensure segmentation mask has same spatial size as image."""
    if seg.shape[0] == h and seg.shape[1] == w:
        return seg
    return mmcv.imresize(seg, (w, h), interpolation='nearest')


@PIPELINES.register_module()
class RandomCropPadAtRandomLocation(object):
    """Random crop + random-location pad to fixed square size.

    This reproduces the behavior of `random_crop_or_pad` from the standalone
    training code:
      - If image/mask is larger than output size, randomly crop.
      - If cropped area is smaller than output size, paste it at a random
        location on a canvas.
      - Padded segmentation pixels are filled with `seg_pad_val` (ignore).
    """

    def __init__(self, crop_size=(512, 512), pad_val=0, seg_pad_val=255):
        assert isinstance(crop_size, tuple) and len(crop_size) == 2
        self.crop_size = crop_size
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def __call__(self, results):
        img = results['img']
        h, w = img.shape[:2]
        out_h, out_w = self.crop_size

        top = random.randint(0, h - out_h) if h > out_h else 0
        left = random.randint(0, w - out_w) if w > out_w else 0
        bottom = min(top + out_h, h)
        right = min(left + out_w, w)

        img_crop = img[top:bottom, left:right, ...]
        crop_h, crop_w = img_crop.shape[:2]

        paste_y = random.randint(0, out_h - crop_h) if crop_h < out_h else 0
        paste_x = random.randint(0, out_w - crop_w) if crop_w < out_w else 0

        if img.ndim == 3:
            img_canvas = np.full((out_h, out_w, img.shape[2]), self.pad_val, dtype=img.dtype)
        else:
            img_canvas = np.full((out_h, out_w), self.pad_val, dtype=img.dtype)
        img_canvas[paste_y:paste_y + crop_h, paste_x:paste_x + crop_w, ...] = img_crop

        results['img'] = img_canvas
        results['img_shape'] = img_canvas.shape
        results['pad_shape'] = img_canvas.shape

        for key in results.get('seg_fields', []):
            seg = _ensure_2d_mask(results[key])
            seg = _align_mask_to_image(seg, h, w)
            seg_crop = seg[top:bottom, left:right]
            seg_crop_h, seg_crop_w = seg_crop.shape[:2]
            seg_canvas = np.full((out_h, out_w), self.seg_pad_val, dtype=seg.dtype)
            seg_canvas[paste_y:paste_y + seg_crop_h, paste_x:paste_x + seg_crop_w] = seg_crop
            results[key] = seg_canvas

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(crop_size={self.crop_size}, '
                f'pad_val={self.pad_val}, seg_pad_val={self.seg_pad_val})')


@PIPELINES.register_module()
class ResizeKeepRatioNoUpscalePad(object):
    """Resize with keep-ratio (downscale only), then center pad to square.

    Mirrors `resize_and_pad_keep_ratio` from the standalone validation dataset:
      - Keep aspect ratio.
      - Never upscale smaller images.
      - Pad with `pad_val`/`seg_pad_val` to `out_size x out_size`.
    """

    def __init__(self, out_size=512, pad_val=0, seg_pad_val=255):
        self.out_size = out_size
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def __call__(self, results):
        img = results['img']
        h, w = img.shape[:2]
        out_size = self.out_size

        scale = min(1.0, float(out_size) / float(max(h, w)))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        if new_w != w or new_h != h:
            img = mmcv.imresize(img, (new_w, new_h), interpolation='bilinear')

        if img.ndim == 3:
            img_canvas = np.full((out_size, out_size, img.shape[2]), self.pad_val, dtype=img.dtype)
        else:
            img_canvas = np.full((out_size, out_size), self.pad_val, dtype=img.dtype)

        pad_x = (out_size - new_w) // 2
        pad_y = (out_size - new_h) // 2
        img_canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w, ...] = img

        results['img'] = img_canvas
        results['img_shape'] = img_canvas.shape
        results['pad_shape'] = img_canvas.shape

        for key in results.get('seg_fields', []):
            seg = _ensure_2d_mask(results[key])
            seg = _align_mask_to_image(seg, h, w)
            if new_w != w or new_h != h:
                seg = mmcv.imresize(seg, (new_w, new_h), interpolation='nearest')
            seg_canvas = np.full((out_size, out_size), self.seg_pad_val, dtype=seg.dtype)
            seg_canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = seg
            results[key] = seg_canvas

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(out_size={self.out_size}, '
                f'pad_val={self.pad_val}, seg_pad_val={self.seg_pad_val})')
