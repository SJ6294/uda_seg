import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IGNORE_INDEX = 255


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_image_file(filename: str) -> bool:
    return any(filename.endswith(ext) for ext in ["jpeg", "JPEG", "jpg", "png", "JPG", "PNG", "gif"])


def _build_img_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _random_horizontal_flip(image: Image.Image, mask: Image.Image, p: float = 0.5) -> Tuple[Image.Image, Image.Image]:
    if random.random() < p:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return image, mask


def _random_rotate(image: Image.Image, mask: Image.Image, p: float = 0.2, degrees: int = 15) -> Tuple[Image.Image, Image.Image]:
    if random.random() < p:
        angle = random.randint(-degrees, degrees)
        image = image.rotate(angle, resample=Image.BICUBIC)
        mask = mask.rotate(angle, resample=Image.NEAREST)
    return image, mask


def random_crop_or_pad(
    image: Image.Image,
    mask: Image.Image,
    out_size: int = 512,
) -> Tuple[Image.Image, Image.Image, Tuple[int, int, int, int]]:
    w, h = image.size

    top = random.randint(0, h - out_size) if h > out_size else 0
    left = random.randint(0, w - out_size) if w > out_size else 0

    right = min(left + out_size, w)
    bottom = min(top + out_size, h)

    image_crop = image.crop((left, top, right, bottom))
    mask_crop = mask.crop((left, top, right, bottom))

    crop_w, crop_h = image_crop.size

    canvas_img = Image.new("RGB", (out_size, out_size), color=(0, 0, 0))
    canvas_mask = Image.new("L", (out_size, out_size), color=IGNORE_INDEX)

    paste_x = random.randint(0, out_size - crop_w) if crop_w < out_size else 0
    paste_y = random.randint(0, out_size - crop_h) if crop_h < out_size else 0

    canvas_img.paste(image_crop, (paste_x, paste_y))
    canvas_mask.paste(mask_crop, (paste_x, paste_y))

    valid_box = (paste_x, paste_y, paste_x + crop_w, paste_y + crop_h)
    return canvas_img, canvas_mask, valid_box


def resize_and_pad_keep_ratio(
    image: Image.Image,
    mask: Image.Image,
    out_size: int = 512,
) -> Tuple[Image.Image, Image.Image, Dict[str, int]]:
    w, h = image.size
    scale = min(1.0, float(out_size) / float(max(h, w)))

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    if new_w != w or new_h != h:
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)
        mask = mask.resize((new_w, new_h), resample=Image.NEAREST)

    canvas_img = Image.new("RGB", (out_size, out_size), color=(0, 0, 0))
    canvas_mask = Image.new("L", (out_size, out_size), color=IGNORE_INDEX)

    pad_x = (out_size - new_w) // 2
    pad_y = (out_size - new_h) // 2

    canvas_img.paste(image, (pad_x, pad_y))
    canvas_mask.paste(mask, (pad_x, pad_y))

    meta = {
        "orig_w": w,
        "orig_h": h,
        "new_w": new_w,
        "new_h": new_h,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "out_size": out_size,
    }
    return canvas_img, canvas_mask, meta


class CamObjDataset(Dataset):
    def __init__(self, data_root: str, size: int = 512):
        self.size = size
        self.img_transform = _build_img_transform()

        images = sorted(os.listdir(os.path.join(data_root, "high")))
        gts = sorted(os.listdir(os.path.join(data_root, "Mask")))

        self.images = [os.path.join(data_root, "high", x) for x in images if is_image_file(x)]
        self.gts = [os.path.join(data_root, "Mask", x) for x in gts if is_image_file(x)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = Image.open(self.images[index]).convert("RGB")
        gt = Image.open(self.gts[index]).convert("L")

        image, gt = _random_horizontal_flip(image, gt, p=0.5)
        image, gt = _random_rotate(image, gt, p=0.2, degrees=15)

        image, gt, valid_box = random_crop_or_pad(image, gt, out_size=self.size)

        image = self.img_transform(image)
        gt = np.array(gt, dtype=np.uint8)

        gt_bin = np.full_like(gt, fill_value=IGNORE_INDEX, dtype=np.int64)
        x1, y1, x2, y2 = valid_box
        valid_region = gt[y1:y2, x1:x2]
        gt_bin[y1:y2, x1:x2] = (valid_region > 0).astype(np.int64)

        gt = torch.from_numpy(gt_bin).long()
        return image, gt


class test_dataset(Dataset):
    def __init__(self, data_root: str, size: int = 512):
        self.size = size
        self.img_transform = _build_img_transform()

        images = sorted(os.listdir(os.path.join(data_root, "high")))
        gts = sorted(os.listdir(os.path.join(data_root, "Mask")))

        self.images = [os.path.join(data_root, "high", x) for x in images if is_image_file(x)]
        self.gts = [os.path.join(data_root, "Mask", x) for x in gts if is_image_file(x)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = Image.open(self.images[index]).convert("RGB")
        gt = Image.open(self.gts[index]).convert("L")

        image, gt, meta = resize_and_pad_keep_ratio(image, gt, out_size=self.size)

        image = self.img_transform(image)
        gt = np.array(gt, dtype=np.uint8)

        gt_bin = np.full_like(gt, fill_value=IGNORE_INDEX, dtype=np.int64)
        x1, y1 = meta["pad_x"], meta["pad_y"]
        x2, y2 = x1 + meta["new_w"], y1 + meta["new_h"]
        valid_region = gt[y1:y2, x1:x2]
        gt_bin[y1:y2, x1:x2] = (valid_region > 0).astype(np.int64)

        gt = torch.from_numpy(gt_bin).long()

        name = os.path.basename(self.images[index])
        if name.endswith(".jpg"):
            name = name[:-4] + ".png"

        return image, gt, name, meta


def get_loader(data_root: str, batchsize: int, size: int, shuffle=None, drop_last: bool = False):
    dataset = CamObjDataset(data_root=data_root, size=size)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return data_loader
