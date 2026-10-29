#!/usr/bin/env python3
"""
bench_subnormal_m205.py
=======================
Quanto costano i numeri SUBNORMALI (< 1e-308) nella valutazione del modello pole-zero + Bessel di
scan_residuals_bessel_m205.py. Nella coda della finestra gli esponenziali dei poli veloci passano
per la fascia subnormale: sui processori x86 del cluster ogni operazione li' costa decine di volte
una normale, sul Mac (ARM) no. EXP_FLOOR = -inf e' il codice di prima, -600 quello corretto.

Gira su un nodo di calcolo:
    echo "source /home/zanelli/LoadOctopus.sh; cd $PWD; OMP_NUM_THREADS=1 python test/fit/bench_subnormal_m205.py" \
        | qsub -q cupid -N BENCH -l walltime=00:20:00 -l mem=1G -o bench.out -e bench.err
"""
import os, sys, time, socket
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import scan_residuals_bessel_m205 as sb

# un fit 10p z6 di ch31 WP5 (criterio time+freq): t0, 6 zeri, 10 poli
THETA = [0.501604, -1635.749344, -1623.275533, -1115.585502, -79.667485, -1662.279334, -12.756975,
         -3275.79229, -3080.424991, -3200.715151, -973.42909, -332.199233, -973.505351, -2990.33588,
         -970.799266, -47.222224, -9.614403]
NZ, N_EVAL = 6, 200

t = (np.arange(10_000) + 0.5) / sb.SAMPLING_RATE
res = {}
for floor in (-np.inf, sb.EXP_FLOOR):
    sb.EXP_FLOOR = floor
    f = sb.make_pulse_bessel(sb.BESSEL_ORDER, sb.FCUT, THETA[1:1 + NZ], THETA[1 + NZ:])
    t0 = time.perf_counter()
    for _ in range(N_EVAL):
        y = f(t - THETA[0])
    res[floor] = ((time.perf_counter() - t0) / N_EVAL * 1e3, y)
(ms_old, y_old), (ms_new, y_new) = res.values()
print(f"{socket.gethostname()}: una valutazione del modello {ms_old:.2f} ms col codice di prima, "
      f"{ms_new:.2f} ms senza subnormali  ->  x{ms_old / ms_new:.1f};  max|differenza| {np.abs(y_old - y_new).max():.1e}")
