# Binary Segmentation Baselines (MAS3K / Deepfish)

This folder provides starter configs for binary semantic segmentation using
2-class softmax + cross-entropy loss (`num_classes=2`).

## Label handling

`LoadAnnotations` now supports binary thresholding in pipeline configs:

- `binary_label=True`
- `binary_threshold=128`

This maps labels as:

- pixel value `< 128` -> class `0` (background)
- pixel value `>= 128` -> class `1` (object)

So noisy labels that are not exactly 0/255 are normalized automatically.

## Data layout expected by configs

These configs use `data_root='data'` and expect:

- `data/MAS3K/train/high`, `data/MAS3K/train/Mask`
- `data/MAS3K/valid/high`, `data/MAS3K/valid/Mask`
- `data/Deepfish/train/high`, `data/Deepfish/train/Mask`
- `data/Deepfish/valid/high`, `data/Deepfish/valid/Mask`

If your dataset is in `B:\3_exp\uda\data`, either copy or symlink it to
`/workspace/uda_seg/seg/data`.

## Provided configs

- `segformer_b5_mas3k_source_only.py`
- `segformer_b5_deepfish_target_only.py`
- `deeplabv3plus_r50_mas3k_source_only.py`
- `deeplabv3plus_r50_deepfish_target_only.py`
- `segformer_b5_mas3k_to_deepfish_uda.py`

## Example commands

```bash
# source-only (MAS3K)
python tools/train.py configs/binary/segformer_b5_mas3k_source_only.py

# target-only (Deepfish)
python tools/train.py configs/binary/segformer_b5_deepfish_target_only.py

# UDA (MAS3K -> Deepfish)
python tools/train.py configs/binary/segformer_b5_mas3k_to_deepfish_uda.py
```
