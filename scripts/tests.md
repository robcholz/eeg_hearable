# Tests

## Extract Yaw Channel

```shell
python scripts/debug_brir_extract.py --brir \
raw_dataset/sbsbrir/SBSBRIR_x1y-1_wav/SBSBRIR_x1y-1_LS90deg.wav \
--yaw 117 --out scripts/brir_stereo.wav
```

## Get Yaw Channel Stats

```shell
scripts/debug_brir_stats.sh scripts/brir_stereo.wav
```

## Convolve

```shell
python scripts/debug_brir_convolve_np.py \
    --input scripts/0.wav \
    --brir scripts/brir_stereo.wav \
    --out scripts/convolved.wav
```

```shell
scripts/debug_brir_gain_sweep.sh scripts/0.wav scripts/brir_stereo.wav
```

## Get Convolved Stats

```shell
scripts/debug_brir_stats.sh scripts/convolved.wav
```