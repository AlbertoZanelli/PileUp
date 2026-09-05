#!/bin/bash
cd /mnt/disk1/data/users/azanelli/PileUp
source /home/zanelli/LoadOctopus.sh
/mnt/disk1/home/zanelli/env/bin/python3 /mnt/disk1/data/users/azanelli/PileUp/analyse_BI_m205_wiener.py --worker --channel 34 --wp 13
