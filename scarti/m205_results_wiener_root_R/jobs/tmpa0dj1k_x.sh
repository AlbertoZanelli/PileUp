#!/bin/bash
cd /mnt/disk1/data/users/azanelli/PileUp
source /home/zanelli/LoadOctopus.sh
/mnt/disk1/home/zanelli/env/bin/python /mnt/disk1/data/users/azanelli/PileUp/analysis_BI_m205_wiener_regolarized_copy.py --worker --channel 91 --wp 15
