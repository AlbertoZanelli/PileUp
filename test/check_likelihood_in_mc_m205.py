"""Controllo dello stimatore M dentro simulate_BI_error_m205.py (LIKELIHOOD = True).

1. Attivare M NON cambia niente per i filtri: con e senza M, le PSD di ogni cartella sono identiche
   bit per bit (M legge gli stessi impulsi DOPO i filtri, senza toccarli).
2. La modalita' confronto resta com'era: ogni cartella simulata insieme alle altre da' le stesse PSD
   che da sola (e' il controllo di test/check_compare_mode_m205.py, qui con M attivo).
3. M sui singoli: il taglio accetta il 90% per costruzione, e il BI di M e' un numero sensato.
NSIM ridotto: ~2 min.

    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 /opt/homebrew/bin/python3.13 test/check_likelihood_in_mc_m205.py
"""
import os, sys, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulate_BI_error_m205 as s

CH, WP, SEED = 31, 7, 1234
s.NSIM = 2000
s.FOLDERS = [s.folder_info("m205_results_wiener_APsimfit10000led_npsclean_swna1"),
             s.folder_info("m205_results_octopus_APsimfit10000led_npsclean")]
rows = [next(r for r in csv.DictReader(open(f["bi_csv"]))
             if int(r["channel"]) == CH and int(r["wp"]) == WP) for f in s.FOLDERS]
lik = s.make_likelihood(CH, WP)

plain = s.run_pair(CH, WP, rows, SEED)
with_m, res_m = s.run_pair(CH, WP, rows, SEED, likelihood=lik)
for f, a, b in zip(s.FOLDERS, plain, with_m):                                        # 1
    assert np.array_equal(a["psd_single"], b["psd_single"]), f["name"]
    assert np.array_equal(a["psd_pileup"], b["psd_pileup"]), f["name"]
print("OK 1: le PSD delle cartelle sono identiche con e senza M")

folders = list(s.FOLDERS)
for f, r, res in zip(folders, rows, with_m):                                          # 2
    s.FOLDERS = [f]
    (alone,), _ = s.run_pair(CH, WP, [r], SEED, likelihood=lik)
    assert np.array_equal(alone["psd_single"], res["psd_single"]), f["name"]
print("OK 2: modalita' confronto invariata con M attivo")

acc = np.mean(res_m["psd_single"] >= res_m["cut"])                                   # 3
assert abs(acc - s.ACCEPTANCE) < 1e-3, acc
assert np.all(np.isfinite(res_m["psd_single"])) and np.all(np.isfinite(res_m["psd_pileup"]))
print(f"OK 3: M accetta il {100*acc:.1f}% dei singoli; BI_mc M {res_m['BI_mc']:.3e} "
      f"vs rapporto dei massimi {with_m[0]['BI_mc']:.3e} ({s.NSIM} eventi: solo un controllo, non una misura)")
