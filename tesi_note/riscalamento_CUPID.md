# Riscalamento delle ampiezze alle condizioni di CUPID (rimandato)

Messo da parte il 2026-10-28: per ora l'analisi si fa nelle condizioni di m205 com'è misurata. Qui c'è tutto quello
che serve per riprenderlo.

## L'idea
L'ampiezza nel light detector è proporzionale a LY × E × G_NTL × (sensibilità dell'NTD). Per portarla alle condizioni
di CUPID si cambiano la luce che arriva al rivelatore e il guadagno NTL, e si tengono NTD e rumore di m205:

    A_CUPID = A_m205 × (G_CUPID / G_ch) × (LY_CUPID / LY_ch)

Siccome il rumore resta quello misurato, il fattore scala l'SNR. I fotoni diventano N = LY_CUPID · E / E_γ, uguali
per tutti i canali (590 a Q_ββ con E_γ = 1.85 eV, 670 nm).

## G_CUPID = 11.3 (Armatol et al. 2026, arXiv:2507.15732, JINST 21 P01035, sez. 5.3)
- Il numero **non è scritto** nell'articolo: si ricava dalle sue equazioni.
- Eq. 5.1: G_CROSS = (1 − ξ_CROSS) + ξ_CROSS · G_NTL, con G_CROSS = 8.66 (media della tab. 4, colonna "Far", cioè
  elettrodi rivolti verso il cristallo) e ξ_CROSS = 0.56. Ne viene G_NTL = 14.7.
- Eq. 5.3 con ξ_CUPID = 0.75 (configurazione III, tab. 5): G_CUPID = 0.25 + 0.75 × 14.7 = 11.3.
- Controllo con l'eq. 5.4: (11.3 / 8.66) × (0.36 / 0.30) = 1.56, il fattore 1.6 della tab. 5.
- Attenzione: nella tab. 4 c'è anche "11.3(2)", ma è il guadagno di LD-5 lontano. È una coincidenza.
- **Il guadagno di Armatol è un guadagno in SNR, non in ampiezza.** Sez. 4.4: "The effective NTL gain G_NTL is defined
  as the improvement in SNR". Lo confermano le tabelle 2 e 4: per LD-5 il G_eff vicino/lontano 4.30 / 11.3 coincide
  con il rapporto dei rumori (72/16.8 = 4.29, 72/6.4 = 11.25), mentre le ampiezze crescono di 5.27 / 13.7.
- Armatol usa fotoni da 2.1 eV.

## LY_CUPID = 0.36 keV/MeV
- Pageot, tesi, cap. 6: "the difference in assembly lead to a different LY (0.3 for CROSS and 0.36 for CUPID)".
  Tab. 7.3: LY 0.3 → scaling 0.83; 0.36 = riferimento.
- Il fattore 1.6 = (0.36/0.30) per la geometria × (circa 1.3) per la copertura degli elettrodi (56% → 75%).
- L'Excel del gruppo (`m205_load_curves/2026 CROSS pile-up Analysis Alberto.xlsx`, colonna J "CUPID, LY scaling")
  usa 0.3: I × 0.3/LY. Il relatore ha confermato **0.36**.

## LY dei nostri canali (prima dell'NTL, LMO vicino)
Foglio "Denys Comparison" di `~/Downloads/CROSS_data_analysis/Octopus/ResultsCaraterization.xlsx` (e CROSS status
16/04/2026, p. 7):

| ch | rivelatore | LY Denys | LY Octopus |
|---|---|---|---|
| 31 | SiNTL_c3 (Si) | 0.17 ± 0.09 | 0.18 ± 0.08 |
| 34 | SiNTL_c5 (Si) | 0.18 ± 0.05 | 0.18 ± 0.05 |
| 71 | SiNTL_c6 (Si) | 0.17 ± 0.05 | 0.16 ± 0.05 |
| 83 | AlSi1 (Si) | 0.20 ± 0.08 | 0.15 ± 0.13 |
| 91 | GeSp_c16 (Ge) | 0.22 ± 0.08 | 0.21 ± 0.08 |

## G_ch: quale guadagno
Deve essere lo **stesso tipo** di G_CUPID, cioè un guadagno in SNR: SNR con NTL diviso SNR senza NTL, realizzato nelle
condizioni di m205.

Tensioni NTL (foglio NTL di `c2u_Run14_monitoring.xlsx`, logbook RUN14):
- run 96 (dicembre 2025): 100 V principale → 100 V per ch31 e ch34, 67 V per ch71, 47 V per ch83 e ch91;
- da m199: **120 V** per tutti e cinque. Run 204–206 con i WP della configurazione #97. La run 205 (la load curve) ha
  PGA gain 1 invece di 5. L'intervento sul criostato è prima della run 189.

Foglio "Noise Comparison" di `ResultsCaraterization.xlsx` (Roberto, Octopus). Colonne C, D, I = rumore [µV] delle
run 92 (0 V), 96, 206. Colonne E, G, J = ampiezza al tallio sul LMO in basso (lato NTL = LMO vicino) [mV] delle stesse
run. Colonna O "eff gain" = SNR96 / SNR92.

| ch | G ampiezza 96/92 | G ampiezza 206/92 (120 V) | **G SNR 96/92** (col. O) | **G SNR 206/92** (120 V) | G SNR Argonauts m199–204 | rumore 206/92 |
|---|---|---|---|---|---|---|
| 31 | 30.6 | 38.8 | 32.0 | 14.0 | 11.4 | 2.78 |
| 34 | 19.0 | 36.0 | 19.9 | 32.3 | 32.5 | 1.11 |
| 71 | 18.8 | 37.0 | 18.5 | 18.3 | 14.8 | 2.02 |
| 83 | 10.0 | 39.6 | 9.4 | 23.8 | 32.5 | 1.66 |
| 91 | 5.6 | 7.0 | 4.7 | 5.2 | 6.3 | 1.35 |

- G SNR 206/92 = (J/I) / (E/C).
- Argonauts: `~/Downloads/CROSS_data_analysis/Argonauts/c2u_Run14_OF_results.xlsx`, foglio NTL, blocco "m199to204",
  colonna BV "NTL gain effective, Close LMO". Nello stesso foglio c'è anche il guadagno in ampiezza (colonna AS:
  37.0, 35.5, 35.1, 31.7, 6.3) e il guadagno sul LED a 120 V (84.7, 72.5, 71.6, 62.8, 9.8).
- Nel codice c'è ancora `G_LMO` = guadagno in **ampiezza** ricavato dal LED (39.1, 34.1, 34.5, 28.1, 7.0): **va
  sostituito** con un guadagno in SNR.

## Fattori risultanti, f = 11.3 × 0.36 / (G_ch × LY_ch)

| ch | con G in ampiezza (codice) | con G SNR 206/92 (Octopus) | con G SNR Argonauts |
|---|---|---|---|
| 31 | 0.61 | 1.71 | 2.10 |
| 34 | 0.66 | 0.70 | 0.70 |
| 71 | 0.69 | 1.31 | 1.62 |
| 83 | 0.72 | 0.86 | 0.63 |
| 91 | 2.64 | 3.57 | 2.95 |

Con il guadagno in SNR, ch31 e ch71 (rumore 2–3 volte più alto a 120 V) passano a un fattore maggiore di 1: la
proiezione assume che CUPID raggiunga il guadagno SNR di Armatol e toglie l'eccesso di rumore visto a 120 V.

## Domande aperte
1. Il guadagno SNR giusto è quello della run 206 (120 V, PGA 5) o va misurato sui dati della 205 (PGA 1)? Deve essere
   quello realizzato nei nostri dati.
2. Octopus o Argonauts? Differiscono soprattutto per ch31, ch71 e ch83.
3. Che la forma della luce del LMO sia la stessa degli impulsi LED (template e rise time vengono dai LED): confronto
   con l'impulso medio dei light detector in coincidenza con la riga a 2615 keV (m204).

## Altre cose verificate per strada
- L'ampiezza nella ROI usa già il guadagno degli eventi veri: A_ROI = A_LED(m205) × A_2615(m204) · 3034/2615 /
  A_LED(m204) (Excel, colonne H e F; `AMP_SCALE_FACTORS` in `risetime_and_amplitude_.py`).
- Nei Si il guadagno NTL sugli impulsi LED è circa il doppio di quello sulla luce del LMO (status p. 11).

## Stato del codice (spento, pronto da riaccendere)
- `utility/functions.py`: `LY_CUPID`, `G_CUPID`, `G_LMO` (da sostituire), `cupid_factor`, `n_photons(..., cupid=True)`.
- `analyse_BI_m205.py`, `analysis_BI_m205_wiener_regolarized.py`: `CUPID = False`. Con True l'ampiezza si moltiplica
  per `cupid_factor` e la cartella prende `_cupid`.
- `simulate_BI_error_m205.py`: se la cartella finisce in `_cupid` usa i fotoni di CUPID.
- `compare_templates_m205.py`: riconosce il suffisso `_cupid`.

## Le celle del notebook tolte (sezione 1.3 "Le condizioni di CUPID")
Markdown: la formula sopra, le voci su G_CUPID, LY_CUPID, G_ch, LY_ch.

Codice della tabella:

```python
show_block(fn, "# ── Proiezione alle condizioni di CUPID", "return G_CUPID")
print(" ch   G_ch   LY_ch     fattore G        fattore LY       totale")
for ch in CHANNELS:
    g, ly = fn.G_LMO[ch], fn.LY_KEV_MEV[ch]
    print(f"{ch:3d}  {g:5.1f}   {ly:.2f}   11.3/{g:4.1f} = {fn.G_CUPID / g:.2f}   0.36/{ly:.2f} = {fn.LY_CUPID / ly:.2f}   {fn.cupid_factor(ch):.2f}")
```

Ampiezza e fotoni del punto:

```python
A_SIG   = A_ROI * fn.cupid_factor(CH) if CUPID else A_ROI     # [V] l'ampiezza che si simula
N_GAMMA = fn.n_photons(CH, cupid=CUPID)                         # fotoni assorbiti a Q_bb
```

Nel grafico di ampiezza e SNR di tutti i punti, le curve CUPID erano `a * fn.cupid_factor(ch)` e
`snr * fn.cupid_factor(ch)`, tratteggiate.
