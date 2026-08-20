#!/bin/bash
cd /mnt/disk1/data/users/azanelli/PileUp
source /home/zanelli/LoadOctopus.sh
/mnt/disk1/home/zanelli/env/bin/python /mnt/disk1/data/users/azanelli/PileUp/scan_residuals_bessel_m205.py --worker --wp 15 --nreal 7 --cc 0 --nzer 3
