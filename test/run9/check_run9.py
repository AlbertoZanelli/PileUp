"""
check_run9.py
=============
Prova analyse_RUN9.py senza il .bin vero: scrive in una cartella temporanea un .bin sintetico del canale 5
(stesso header e codifica a 24 bit; rumore gaussiano con la NPS di Pageot nelle finestre dei .pos, 0 V
altrove) e fa girare run_channel su N eventi. Controlla che la NPS misurata torni (nps_ratio ~ 1) e che
M e Y diano un rigetto sensato. Il rumore e' gaussiano e stazionario: i numeri NON sono quelli veri.

    KMP_DUPLICATE_LIB_OK=TRUE python3 test/run9/check_run9.py      # ~15 s
"""
import os
import sys
import struct
import tempfile

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import analyse_RUN9 as A

CH, N, ADC = 5, 1000, 10.0

starts = (np.loadtxt(A.pos_path("pup_n1-d0", CH))[:N, 0] // 4).astype(int)
assert np.array_equal(starts, (np.loadtxt(A.pos_path("pup_n1-d8", CH))[:N, 0] // 4).astype(int))   # stesse finestre
nps = np.load(os.path.join(A.DATA_DIR, "NPS", f"NPS_channel{CH}_win{A.WINDOW}_{A.MEAS}.npy")) * 8 / 3

words = np.full(starts.max() + A.WINDOW + 8, (2 ** 23) << 8, dtype=np.uint32)      # 0 V ovunque
rng = np.random.default_rng(1)
for s in starts:          # la lettura salta la prima parola del file: la finestra s parte dalla parola s+1
    v = np.fft.ifft(np.sqrt(nps) * (rng.normal(size=A.WINDOW) + 1j * rng.normal(size=A.WINDOW))).real
    words[s + 1:s + 1 + A.WINDOW] = np.round((v + ADC) / ADC * 2 ** 23).astype(np.uint32) << 8
words[:3] = np.frombuffer(struct.pack("<c3xff", b"<", 10000.0, ADC), dtype=np.uint32)

with tempfile.TemporaryDirectory() as tmp:
    words.tofile(os.path.join(tmp, f"{A.MEAS}_{CH:03}_000.bin"))
    res = A.run_channel(CH, tmp, N, tmp)
    z = np.load(os.path.join(tmp, f"discriminants_ch{CH}.npz"))
    assert all(len(z[k]) == N for k in z.files)
print({k: round(v, 4) for k, v in res.items()})
assert abs(res["nps_ratio"] - 1) < 0.05, "la NPS misurata non torna con quella usata"
assert 0.5 < res["rp_M"] < 1 and 0.5 < res["rp_Y"] < 1, "rigetto fuori scala"
print("OK")
