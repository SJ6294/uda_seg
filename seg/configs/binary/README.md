# Binary Segmentation Baselines (MAS3K / Deepfish)

This folder provides starter configs for binary semantic segmentation using
2-class softmax + cross-entropy loss (`num_classes=2`).

## Label handling

`LoadAnnotations` supports binary thresholding in pipeline configs:

- `binary_label=True`
- `binary_threshold=128`

This maps labels as:

- pixel value `< 128` -> class `0` (background)
- pixel value `>= 128` -> class `1` (object)

So noisy labels that are not exactly 0/255 are normalized automatically.

## Train-time spatial preprocessing

For train pipelines in these configs:

- **No resize is used**.
- `RandomCrop(crop_size=(512, 512))` is applied directly to original images.
- `Pad(size=(512, 512), seg_pad_val=255)` is applied after crop.

Therefore if an image/mask is smaller than 512 in any dimension, padded mask
pixels are treated as **ignore label (255)**, so they are excluded from loss and mIoU.

## Data layout expected by configs

These configs use `data_root='data'` and expect:

- `data/MAS3K/train/high`, `data/MAS3K/train/Mask`
- `data/MAS3K/valid/high`, `data/MAS3K/valid/Mask`
- `data/Deepfish/train/high`, `data/Deepfish/train/Mask`
- `data/Deepfish/valid/high`, `data/Deepfish/valid/Mask`

Image suffix is `.jpg` and label suffix is `.png`.

If your dataset is in `B:\3_exp\uda\data`, either copy or symlink it to
`/workspace/uda_seg/seg/data`.


## Supervised vs UDA configs

- `*_source_only.py` and `*_target_only.py` are now **pure supervised** configs (no `uda` block, no DACS wrapper).
- `segformer_b5_mas3k_to_deepfish_uda.py` is the only UDA config in this folder.

## Provided configs

- `segformer_b5_mas3k_source_only.py`
- `segformer_b5_deepfish_target_only.py`
- `deeplabv3plus_r50_mas3k_source_only.py`
- `deeplabv3plus_r50_deepfish_target_only.py`
- `segformer_b5_mas3k_to_deepfish_uda.py`


## Validation preprocessing

- Validation pipeline pads with `size_divisor=32` (not fixed 512) to avoid OpenCV pad errors on larger images.
- Padding label uses `255` so padded regions are ignored in loss and mIoU.

## Validation outputs and logging

During validation, the trainer now saves concatenated visualization images to
`<work_dir>/result/` in the order:

- original image
- prediction (black/white)
- ground truth (black/white)

Validation logs include:

- `IoU.background`
- `IoU.object`
- `mIoU`
- `val_loss`

Training logs continue to include `loss` (train loss).

## Batch setting

- `samples_per_gpu=2` (so with 1 GPU, effective train batch size is 2).

## Example commands (single GPU)

```bash
# source-only (MAS3K)
python tools/train.py configs/binary/segformer_b5_mas3k_source_only.py --gpu-id 0

# target-only (Deepfish)
python tools/train.py configs/binary/segformer_b5_deepfish_target_only.py --gpu-id 0

# UDA (MAS3K -> Deepfish)
python tools/train.py configs/binary/segformer_b5_mas3k_to_deepfish_uda.py --gpu-id 0
```
