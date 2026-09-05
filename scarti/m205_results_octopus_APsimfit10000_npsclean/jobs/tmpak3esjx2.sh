#!/bin/bash
cd /mnt/disk1/data/users/azanelli/PileUp
source /home/zanelli/LoadOctopus.sh
/mnt/disk1/home/zanelli/env/bin/python3 /mnt/disk1/data/users/azanelli/PileUp/m205_results_octopus_APsimfit10000_npsclean/_analyse_BI_m205.py --worker --channel 91 --wp 3
