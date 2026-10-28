# Richiesta figure — sezione 4.4 "Trainable Wiener Filter" della tesi

Scritta il 06/10/2026 dalla chat della tesi per la chat delle analisi (questa cartella). Sostituisce
la richiesta precedente dello stesso giorno, che Alberto aveva bocciato. Con Alberto si parla in
**italiano**; il testo delle figure è in **inglese**.

**Il testo della sezione è già scritto**: `../MasterThesis___Alberto_Zanelli/Chapter4/chapter4.tex`,
`\section{Trainable Wiener Filter}` (righe ~1183–1540). Le figure sono già richiamate con i nomi
qui sotto e, finché il file non esiste, LaTeX mostra un riquadro "Figure in preparation". Basta
copiare i PDF con il nome giusto in `../MasterThesis___Alberto_Zanelli/Chapter4/figures/` e
ricompilare: non serve toccare il `.tex`. Leggere la sezione prima di fare le figure: le didascalie
sono già scritte e le figure devono mostrare quello che dicono.

Regole di questa cartella (vedi `HANDOFF.md`): niente addestramenti in locale; Alberto committa
lui; nessuna riga "Co-Authored-By: Claude".

## Che cosa racconta la sezione (per scegliere bene le figure)
1. **Kernel di Wiener** W_λ = S*/(|S|² + λN) (spettri normalizzati come in CUORE, `compute_W_torch`):
   λ → ∞ forma del filtro ottimo, λ → 0 deconvoluzione pura. λ si addestra con m₁, m₂.
2. **Senza regolarizzazione il Wiener si sovra-addestra.** Catena causale scritta nel testo:
   punto difficile (BI dell'algoritmo originale alto) → λ crolla → kernel quasi deconvoluzione →
   rumore amplificato, s_i = σ_i/A grandi, fuori dalle ipotesi del modello analitico → il modello
   ottiene comunque una σ_Y,0 piccola perché i due filtri diventano quasi identici (ρ → 1) →
   BI analitico falsamente basso → sovra-addestramento. Negli eventi simulati la cancellazione
   non avviene: il massimo cade su un picco di rumore.
3. **Penalità** w(s₁² + s₂²), w = 1, solo nell'addestramento. Curve di addestramento: il costo si
   appiattisce entro 100–150 passi, λ continua a crescere (direzione piatta: i pesi compensano λ).
4. **Risultato**: Wiener con penalità contro algoritmo originale, nessun guadagno; stessa famiglia
   di filtri (W/H reale e positivo).

⚠ **Il crollo di λ NON segue l'SNR_OF** (Alberto aveva chiesto "λ in funzione dell'SNR"): su 75
punti la correlazione fra log λ e log SNR_OF è 0.07, ch31 (SNR 20–25) non crolla mai, ch34 crolla
a SNR 40–97 ai bias bassi ma non a SNR 90 a 40 V. Segue invece il **BI dell'algoritmo originale**
(correlazione −0.59 su tutti i punti, da −0.64 a −0.91 dentro ch34/71/83/91). Per questo il
pannello (b) della figura 2 ha in ascissa il BI dell'OF. Se Alberto vuole comunque λ contro SNR_OF,
farla come figura di controllo e mostrargliela: il testo dice esplicitamente che l'SNR non separa
i due gruppi.

## Stile (uguale alle figure già nel cap. 4)
- Come `plot_thesis_loadcurve_m205.py` e `plot_thesis_hallucination_m205.py`: matplotlib con
  `text.usetex`, serif 10 pt, tacche interne su tutti e quattro i lati, PDF, `savefig.bbox="tight"`.
  Larghezza piena: `(5.9, 2.6)` in per due pannelli affiancati; `(5.9, 4.8)` per 2×2; `(5.9, 5.2)`
  per 3×2 (la `\textwidth` della tesi è 5.71 in).
- Colori e marker dei canali come nelle figure della curva di carico (`CH_COLOR`): 31 `#2a78d6` ●,
  34 `#eb6834` ■, 71 `#1baf7a` ◆, 83 `#eda100` ▲, 91 `#e87ba4` ▼. Asse V_bias logaritmico con tacche
  0.6, 1, 2, 5, 10, 20, 40. Legende con "ch. 31" ecc. (il cap. 4 nomina i canali).
- Barre d'errore = **larghezza** (P84 − P16)/2 dei 50 semi, sui ΔBI appaiati seme per seme
  (ΔBI = mediana di BI_A/BI_B − 1, in %), come `compare_templates_m205.py` con `ERROR_BAR = "width"`.
- Niente titoli dentro la figura; pannelli marcati (a), (b), ... in alto a sinistra.
- Uno script unico, `plot_thesis_wiener_m205.py`, che **legge i file esistenti senza ricalcolare
  addestramenti**, stampa i numeri citati nel testo e ha `assert` (con tolleranza) sui numeri
  qui sotto. Output in `thesis_figures/wiener/`, poi copia in
  `../MasterThesis___Alberto_Zanelli/Chapter4/figures/`.

## Cartelle dei dati
- OF (algoritmo originale): `m205_results_octopus_APsimfit10000led_npsclean`
- Wiener **senza** penalità: `m205_results_wiener_APsimfit10000led_npsclean`
- Wiener **con** penalità (algoritmo finale, "Wwna"): `m205_results_wiener_APsimfit10000led_npsclean_swna1`
- MC: `BI_mc_error_m205_seeds50.csv` in ogni cartella, 50 semi, stessi eventi (appaiati).
- Filtri: `trained_filters/{f1,f2,kernel}_ch*_wp*.npy` (metà indipendente, 5001 bin; `full_spectrum`
  di `simulate_BI_error_m205.py` per lo spettro intero). Storia: `training_history/hist_ch*_wp*.npz`
  (chiavi `J`, `lam`; in `scarti/..._swna1_hist` anche `loss`, `s1`, `s2`).
- Template di addestramento: AP simulato `APsimfit10000led`; eventi MC generati dal fit; NPS clean.

## Le figure (nome file = quello già nel `.tex`)

### 1. `wiener_kernel.pdf` — che cosa fa λ (fig:pu_wiener_kernel), (5.9, 2.6)
Da `explain_wiener_lambda_m205.py`, **ch34 WP3** (1 V, SNR_OF = 61), ma con **λ = 1 esatto** (il PNG
vecchio usa 1.1) e λ = 10⁻⁴.
- (a) |S|² e λ·N nella normalizzazione di `compute_W_torch`, per λ = 1 e 10⁻⁴, log-log, 1 Hz–5 kHz;
  linee verticali punteggiate alle frequenze di incrocio |S|² = λN, con l'etichetta in Hz.
- (b) impulso **senza rumore** dopo il filtro ottimo, dopo W₁ e dopo W_{10⁻⁴}, normalizzati al
  massimo, tempo in ms (circa −5…+15 ms). In legenda la deviazione standard del rumore filtrato
  relativa all'ampiezza (= s_i con m_i = 1; per l'OF deve venire 1/SNR_OF ≈ 0.016).
- Numeri nel testo, scritti per λ = 1.1: incroci ~140 Hz e ~860 Hz; rumore/ampiezza 0.016 (OF),
  0.07, ~2. **Stampare i valori per λ = 1** e scriverli nell'handoff: il testo va corretto se cambiano.

### 2. `wiener_nopen.pdf` — anatomia del sovra-addestramento (fig:pu_wiener_nopen), 2×2 (5.9, 4.8)
Tutto dal Wiener **senza** penalità, 75 punti, contro l'OF sugli stessi eventi.
- (a) ΔBI Wiener senza penalità − OF (%), **analitico** (marker vuoti) e **Monte Carlo** (pieni,
  appaiati, barre = larghezza), in funzione di V_bias, **solo ch34 e ch91**. Asse y lineare (circa
  −30…+45%), linea a zero.
- (b) λ finale in funzione del BI MC dell'OF (mediana dei 50 semi), log-log, 75 punti, colori dei
  canali; linea orizzontale punteggiata a λ = 0.1 (la soglia usata nel testo) e a λ = 1.
- (c) s₁ in funzione di λ finale, log-log, 75 punti.
- (d) 1 − ρ in funzione di s₁, log-log, 75 punti, ρ = σ₁₂/(σ₁σ₂); curve tratteggiate a σ_Y,0
  costante per s₁ = s₂, cioè 1 − ρ = σ²/(2 s₁²), per σ = 0.01, 0.02, 0.05 (etichettate).
- **s₁, s₂ e ρ vanno ricalcolati con il codice di produzione** (`an.compute_vars_wiener` con il
  kernel salvato, l'AP simulato e la NPS clean), non con scorciatoie. Attenzione al bug del
  salvataggio: in questa cartella il kernel è salvato con un passo di λ di ritardo rispetto a f₁,
  f₂ (vedi `HANDOFF.md`, "Saving bug"); controllare che mean|f·W·S| ≈ 1 e dirlo se non torna.
  `diagnose_lambda_collapse_m205.py --diag` fa già questi conti (config: `RESULTS_NAME` = la
  cartella senza penalità, `GEN_TEMPLATE = "fit"`, `OUT_DIR` nuovo); ~10 s per punto.
- **Controllo diretto della cancellazione** (serve per una frase del testo, nessun addestramento):
  per i 75 punti senza penalità, σ_Y,0 analitica contro la larghezza misurata della Y dei singoli
  simulati (robusta, (P84−P16)/2), lo stesso con la penalità. Atteso: senza penalità, dove λ < 0.1,
  la σ_Y,0 analitica è molto più piccola di quella misurata; con la penalità le due coincidono
  (in `HANDOFF.md` §5 questo rapporto è chiamato "acceptance test"). Riportare i numeri; non
  serve una figura.
- Numeri del testo da confermare (calcolati dalla chat della tesi il 06/10 sui CSV e, per ρ, con
  numpy sui filtri salvati):
  - λ < 0.1 in **15/75** punti, minimo 1.0×10⁻⁷ (ch34 WP1); λ < 1 in 33/75; mediana 1.3;
  - BI MC dell'OF: mediana **1.5×10⁻⁴** dove λ < 0.1, **7.2×10⁻⁵** negli altri punti; SNR_OF
    mediano **36.5 e 37.5** (cioè non separa); ch31: λ fra 1.0 e 6.3;
  - dove λ < 0.1: s₁ fra 0.10 e 2.3 (e 63 su ch34 WP1), contro 0.06 di mediana negli altri;
    1 − ρ mediano **6×10⁻⁴** contro **0.05**; σ_Y,0 analitica mediana **0.022** contro **0.021**;
  - ch91 con λ di qualche unità (WP13–21): s₁ = 0.15–0.18;
  - ΔBI analitico: migliore dell'OF in **25/75** punti, fino a **−25%** (ch34 WP1);
  - ΔBI MC: peggiore in **75/75**, mediana **+1.05%**, ch91 mediana **+13%**, massimo **+41%**;
  - ch34 WP3 (1 V): analitico **−9.0%**, MC **+36%**; BI_MC/BI_an massimo **1.64** (ch91 WP27).

### 3. `wiener_events.pdf` — un evento filtrato (fig:pu_wiener_events), 3×2 (5.9, 5.2)
Da `plot_filtered_examples_m205.py`, **ch34 WP3, r = 0.5, Δt = 0.8 ms**, stesso rumore nelle due
colonne. Righe: OF; Wiener senza penalità (λ = 9.6×10⁻⁵); Wiener con penalità (λ = 1.05). Colonne:
singolo, pile-up. Linea grigia = evento prima dei filtri; due colori per i due filtri; in un
angolo X₁/A, X₂/A e Y. Tempo in ms (circa −10…+40). La riga centrale ha una scala verticale
diversa (la didascalia lo dice). Niente titoli: la didascalia spiega righe e colonne.
- Numeri del testo da confermare e stampare: frazione di pile-up con r = 0.5, Δt = 0.8 ms
  rigettati al taglio del 90% dei singoli: **82% (OF), 6% (senza penalità), 80% (con)**; nel
  pannello centrale X_i ≈ 4.7 A (il testo dice "circa cinque volte l'impulso" e s_i ≈ 2).

### 4. `wiener_training.pdf` — curve di addestramento con penalità (fig:pu_wiener_training), (5.9, 2.6)
Da `..._swna1/training_history/hist_ch*_wp*.npz`, tre punti: **ch31 WP7 (1.8 V), ch34 WP15 (5 V),
ch91 WP15 (5 V)**.
- (a) ⟨M⟩ (= `J`) normalizzato al suo valore finale, in funzione del passo (asse y log se serve).
- (b) λ in funzione del passo, asse y log.
- Numeri del testo da confermare (da `HANDOFF.md`, 30/09, ricalcolarli dai npz): costo piatto dopo
  100–150 passi; λ finale mediana **7.6**, > 1 in **71/75**, massimo **776** (ch91), minimo 0.78;
  λ ancora in crescita negli ultimi 100 passi in **63/75**; variazione del costo negli ultimi 100
  passi **0.09%** in mediana. Se possibile, dalla storia di `scarti/..._swna1_hist` (che ha `loss`),
  la quota della penalità nella funzione di costo: il testo dice **0.4–2%**.

### 5. `wiener_vs_of.pdf` — risultato (fig:pu_wiener_vs_of), (5.9, 2.6), un pannello
ΔBI MC Wiener con penalità − OF (%) in funzione di V_bias, 5 canali, barre = larghezza, linea a
zero. Numeri del testo (verificati sui CSV dalla chat della tesi, rimetterli negli `assert`):
mediana **+0.30%**; Wiener migliore in **11/75**, mai oltre due larghezze; peggiore oltre due
larghezze in **28**; mediane per canale **+0.24 / +0.31 / +0.29 / +0.42 / +0.06%**
(ch31/34/71/83/91). Rapporti BI_MC/BI_an: con penalità **1.003–1.12**, OF **1.02–1.13**; s₁ con
penalità sempre < 0.1.

## Altri numeri del testo da confermare (senza figura)
- "Due addestramenti identici del Wiener differiscono fino al 2.7% su un punto, 0.00% in mediana"
  (`_swna1` contro `scarti/..._swna1_hist`): dire se è il BI MC o l'analitico, e su quali eventi.
- Famiglia di filtri: 45/45 punti con lo stesso costo entro 2×10⁻⁴; λ riscalato di ×0.1…×100 con
  i pesi compensati: costo invariato entro 10⁻⁶. Sono in `HANDOFF.md`; non rifarli (lo script
  `test/check_family_reach_m205.py` non gira più), solo segnalare se qualcosa li contraddice.

## Consegna
- PDF in `thesis_figures/wiener/` e copiati in `../MasterThesis___Alberto_Zanelli/Chapter4/figures/`
  con i cinque nomi sopra.
- Una sezione "Wiener" in `../MasterThesis___Alberto_Zanelli/HANDOFF_figure_curva_carico.md` con:
  che cosa mostra ogni figura, i numeri letti dai dati, **ogni numero che non coincide con quelli
  qui sopra** (scriverlo invece di adattarlo), cautele.
- Mostrare le figure ad Alberto prima di chiudere.
