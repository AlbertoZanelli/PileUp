#!/bin/bash
cd /mnt/disk1/data/users/azanelli/PileUp
source /home/zanelli/LoadOctopus.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/disk1/home/zanelli/env/bin/python /mnt/disk1/data/users/azanelli/PileUp/scan_residuals_bessel_m205.py --worker --channel 91 --wp 13 --nreal 6 --cc 0 --nzer 2
