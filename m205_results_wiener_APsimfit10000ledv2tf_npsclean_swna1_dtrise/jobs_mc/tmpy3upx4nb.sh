#!/bin/bash
cd /mnt/disk1/data/users/azanelli/PileUp
source /home/zanelli/LoadOctopus.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/disk1/home/zanelli/env/bin/python /mnt/disk1/data/users/azanelli/PileUp/m205_results_wiener_APsimfit10000ledv2tf_npsclean_swna1_dtrise/_simulate_BI_error_m205.py --worker --channel 31 --wp 13
