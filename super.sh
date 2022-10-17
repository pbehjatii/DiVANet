python train.py --upsample_scale 3 --scale 2 --style avg --resume

python train.py --upsample_scale 3 --scale 2 --style bicubic --resume

# ------------------------------------------------------------------

python train.py --upsample_scale 4 --scale 3 --style avg --resume

python train.py --upsample_scale 4 --scale 3 --style bicubic --resume

python train.py --upsample_scale 4 --scale 3 --style bicubic --gaussian --resume

# -----------------------------------------------------------------

python train.py --upsample_scale 5 --scale 4 --style avg --resume

python train.py --upsample_scale 5 --scale 4 --style bicubic --resume

python train.py --upsample_scale 5 --scale 4 --style bicubic --gaussian --resume

#----------------------------------------------------------------

python train.py --upsample_scale 2 --scale 2 --style xtox

python train.py --upsample_scale 3 --scale 3 --style xtox

python train.py --upsample_scale 4 --scale 4 --style xtox
