"""Controllo della modalita' confronto (COMPARE) di simulate_BI_error_m205.py.

Ogni cartella simulata INSIEME alle altre deve dare PSD identiche, bit per bit, alla stessa
cartella simulata da sola con lo stesso seed. Se no, un filtro tocca gli impulsi di quello dopo
e gli eventi non sono piu' gli stessi. NSIM ridotto: ~1 min.

    KMP_DUPLICATE_LIB_OK=TRUE /opt/anaconda3/envs/pyrootAlbi/bin/python test/check_compare_mode_m205.py
"""
import os, sys, csv, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulate_BI_error_m205 as s

CH, WP, SEED = 31, 1, 1234
s.NSIM = 2000
assert len(s.FOLDERS) > 1, "COMPARE e' vuota: niente da confrontare"
rows = [next(r for r in csv.DictReader(open(f["bi_csv"]))
             if int(r["channel"]) == CH and int(r["wp"]) == WP) for f in s.FOLDERS]

t = time.time()
together = s.run_pair(CH, WP, rows, SEED)
t_all, t_alone = time.time() - t, 0.0
for f, r, res in zip(list(s.FOLDERS), rows, together):
    s.FOLDERS = [f]
    t = time.time()
    (alone,) = s.run_pair(CH, WP, [r], SEED)
    t_alone += time.time() - t
    assert np.array_equal(alone["psd_single"], res["psd_single"]), f["name"]
    assert np.array_equal(alone["psd_pileup"], res["psd_pileup"]), f["name"]
    print(f"OK {f['name']}: BI_mc {res['BI_mc']:.4e}, identico da sola e insieme")
print(f"tempo: insieme {t_all:.1f} s, una cartella alla volta {t_alone:.1f} s")
