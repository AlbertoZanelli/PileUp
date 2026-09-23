#!/bin/bash
cd /mnt/disk1/data/users/azanelli/PileUp
source /home/zanelli/LoadOctopus.sh
/mnt/disk1/home/zanelli/env/bin/python3 /mnt/disk1/data/users/azanelli/PileUp/m205_results_wiener_APsimfit10000_npsclean_swna1/_analysis_BI_m205_wiener_regolarized.py --worker --channel 34 --wp 23
