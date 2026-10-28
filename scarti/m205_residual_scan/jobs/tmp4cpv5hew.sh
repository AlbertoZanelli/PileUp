#!/bin/bash
cd /Users/albertozanelli/Desktop/Tesi_Erasmus/PileUp
source /home/zanelli/LoadOctopus.sh
/opt/homebrew/opt/python@3.13/bin/python3.13 /Users/albertozanelli/Desktop/Tesi_Erasmus/PileUp/scan_residuals_m205.py --worker --wp 1 --nreal 3 --cc 0 --nzer 0
