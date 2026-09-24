"""
check_mc_progress_m205.py
=========================
A che punto e' il Monte Carlo lanciato con simulate_BI_error_m205.py, e quanto manca.

Per ogni punto (canale, WP) legge:
  - dai CSV del MC quanti seed sono gia' scritti (una riga per seed, per ogni set: le cartelle e,
    con LIKELIHOOD, la cartella di M). I log dei job NON servono: PBS li copia solo a job finito;
  - da `qstat -f` lo stato del job, da quanto gira (walltime) e quanta CPU ha usato (cput).
Da qui: secondi per seed, tempo che manca, e il rapporto CPU/tempo reale, cioe' quanti core il job
sta usando davvero. Il job ne chiede UNO: un rapporto molto sopra 1 vuol dire che numpy/scipy
aprono un thread per core del nodo e i job si rubano la CPU a vicenda.

Legge la configurazione (cartelle, N_SEEDS, canali) da simulate_BI_error_m205.py: deve essere la
stessa del lancio.

    python check_mc_progress_m205.py
"""
import csv
import os
import re
import statistics
import subprocess
from collections import Counter

import simulate_BI_error_m205 as mc

EXPECTED_S_PER_SEED = 220   # misurato in locale: OF + M, 50 000 eventi per popolazione, 2 thread liberi


def seconds(hms):
    h, m, s = (int(v) for v in hms.split(":"))
    return 3600 * h + 60 * m + s


def hours(s):
    return f"{s / 3600:5.1f} h"


def seeds_done(path, only_filter=None):
    """(canale, wp) -> seed gia' scritti nel CSV del MC."""
    done = Counter()
    if not os.path.exists(path):
        return done
    for r in csv.DictReader(open(path)):
        if r.get("gen", mc.GEN_TEMPLATE) == mc.GEN_TEMPLATE and (only_filter is None or r.get("filter") == only_filter):
            done[(int(r["channel"]), int(r["wp"]))] += 1
    return done


def my_jobs():
    """(canale, wp) -> campi di qstat -f dei job di questo MC (nome MC<canale>_<wp>)."""
    try:
        out = subprocess.run(["qstat", "-f"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        print("[WARN] qstat non trovato: mostro solo i seed scritti")
        return {}
    user = os.environ.get("USER", "")
    jobs = {}
    for block in out.split("Job Id:")[1:]:
        f = dict(re.findall(r"^\s+([\w.]+) = (.*)$", block, flags=re.M))
        m = re.fullmatch(rf"{mc.JOB_NAME_PREFIX}(\d+)_(\d+)", f.get("Job_Name", "").strip())
        if m and f.get("Job_Owner", "").startswith(user + "@"):
            jobs[(int(m[1]), int(m[2]))] = f
    return jobs


def main():
    csvs = [(f["name"], os.path.join(f["dir"], mc.OUT_NAME), None) for f in mc.FOLDERS]
    if mc.LIKELIHOOD:
        d = mc.likelihood_dir()
        csvs.append((os.path.basename(d), os.path.join(d, mc.OUT_NAME), "likelihood"))
    counts = [seeds_done(p, flt) for _, p, flt in csvs]
    points = [(int(r["channel"]), int(r["wp"])) for r in mc.select_rows()]
    jobs = my_jobs()

    print(f"MC: {', '.join(n for n, _, _ in csvs)}")
    print(f"    {mc.N_SEEDS} seed x {mc.NSIM} eventi per punto, file {mc.OUT_NAME}\n")
    print(f"{'punto':>10} {'stato':>6} {'seed':>7} {'da':>8} {'s/seed':>7} {'manca':>8} {'CPU/reale':>9}  nodo")
    per_seed, ratio, eta_running, queued, finished = [], [], [], 0, 0
    for p in points:
        done = min(c[p] for c in counts)             # un seed conta quando TUTTI i set l'hanno scritto
        j = jobs.get(p, {})
        state = j.get("job_state", "-")
        wall = seconds(j["resources_used.walltime"]) if "resources_used.walltime" in j else 0
        cpu = seconds(j["resources_used.cput"]) if "resources_used.cput" in j else 0
        sps = wall / done if state == "R" and done else None
        left = (mc.N_SEEDS - done) * sps if sps else None
        r = cpu / wall if wall else None
        if done >= mc.N_SEEDS:
            finished += 1
        elif state == "Q":
            queued += 1
        if sps:
            per_seed.append(sps); eta_running.append(left)
        if r:
            ratio.append(r)
        print(f"{p[0]:>5}/{p[1]:<4} {state:>6} {done:>3}/{mc.N_SEEDS:<3} {hours(wall) if wall else '':>8} "
              f"{f'{sps:.0f}' if sps else '':>7} {hours(left) if left else '':>8} "
              f"{f'{r:.1f}' if r else '':>9}  {j.get('exec_host', '').split('/')[0]}")

    print(f"\nfiniti {finished}/{len(points)}, in coda {queued}, in esecuzione "
          f"{sum(1 for p in points if jobs.get(p, {}).get('job_state') == 'R')}")
    if per_seed:
        med = statistics.median(per_seed)
        print(f"secondi per seed (mediana): {med:.0f}   atteso con 2 core liberi: ~{EXPECTED_S_PER_SEED}  "
              f"-> {med / EXPECTED_S_PER_SEED:.1f} volte piu' lento")
        print(f"i job in esecuzione finiscono fra: {hours(min(eta_running))} (primo) - {hours(max(eta_running))} (ultimo)")
        if queued:
            print(f"quelli in coda, una volta partiti: ~{hours(mc.N_SEEDS * med)} ciascuno")
    elif any(jobs.get(p, {}).get("job_state") == "R" for p in points):
        print("nessun job in esecuzione ha ancora scritto un seed: rilancia fra un po'")
    if ratio:
        rm = statistics.median(ratio)
        print(f"CPU usata / tempo reale (mediana): {rm:.1f} core per job, ne e' stato chiesto 1")
        if rm > 1.5:
            print("  -> ogni job usa piu' core di quelli chiesti: i thread di numpy/scipy di tutti i job\n"
                  "     si contendono i nodi. Rimedio (per i prossimi lanci): OMP_NUM_THREADS=2 nei job\n"
                  "     e 2 core chiesti a PBS (-l nodes=1:ppn=2).")
        elif rm < 0.7:
            print("  -> il job e' spesso fermo (nodo sovraccarico da altri, o in attesa del disco)")
    hosts = Counter(j.get("exec_host", "").split("/")[0] for j in jobs.values() if j.get("job_state") == "R")
    if hosts:
        print("job in esecuzione per nodo: " + ", ".join(f"{h} {n}" for h, n in hosts.most_common()))


if __name__ == "__main__":
    main()
