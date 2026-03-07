# Custom MIC UDA Configs (reconstructed from standalone crop trainer)

This folder rebuilds the standalone `dataset_ce2ch_crop.py` +
`train_sota_segformer_crop.py` behavior inside mmseg/DAFormer UDA.

## Key points

- Supports both **DACS+MIC** and **MinEnt+MIC (without DACS/HRDA)**.
- Uses a custom train transform that reproduces:
  - random horizontal flip (p=0.5)
  - random rotate (p=0.2, degree=15)
  - random crop if larger than 512
  - random-location paste onto a 512 canvas when smaller
- Uses a custom val transform that reproduces:
  - keep-ratio downscale only (no upscaling)
  - center padding to `512x512`
- Padded segmentation labels are `255` (ignore index).
- Validation concat images are saved in the standalone order: `[image | gt | pred]`.

## Configs

- `segformer_b5_mas3k_to_deepfish_mic_sota.py` (DACS + MIC)
- `segformer_b5_mas3k_to_deepfish_minent_mic_sota.py` (MinEnt + MIC, no DACS/HRDA)

## Run

```bash
cd /workspace/uda_seg/seg
# DACS + MIC
python tools/train.py configs/custom_mic/segformer_b5_mas3k_to_deepfish_mic_sota.py --gpu-id 0

# MinEnt + MIC (without DACS/HRDA)
python tools/train.py configs/custom_mic/segformer_b5_mas3k_to_deepfish_minent_mic_sota.py --gpu-id 0
```

## Data layout

Expected under `seg/data`:

- `MAS3K/train/high`, `MAS3K/train/Mask`
- `Deepfish/train/high`, `Deepfish/train/Mask`
- `Deepfish/valid/high`, `Deepfish/valid/Mask`
