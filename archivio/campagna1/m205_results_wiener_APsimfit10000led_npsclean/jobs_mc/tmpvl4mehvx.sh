#!/bin/bash
cd /mnt/disk1/data/users/azanelli/PileUp
source /home/zanelli/LoadOctopus.sh
/mnt/disk1/home/zanelli/env/bin/python3 /mnt/disk1/data/users/azanelli/PileUp/m205_results_wiener_APsimfit10000led_npsclean/_simulate_BI_error_m205.py --worker --channel 91 --wp 1
