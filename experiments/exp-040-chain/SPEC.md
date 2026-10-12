# exp-040-chain — SPEC

Worker `chain`, tier hard, 2026-09-08. Assignment `assignments/q09.md` (Pro queue Q09).
Laptop, 4 threads, no GPU; every model fit runs on Kaggle CPU (`assignments/LAPTOP.md`:
a fit over more than one fold goes to Kaggle). No upload, no candidate without the gate.

## The reading taken

Q09 asks for chain inputs inside a real serving member, the member and its override
reproduced exactly, one matched pair of fits, both propagated through identical full
serving paths, and advancement only on a final-predictor paired benefit. exp-037 §7
established that `gap_arr1..3` are already inside every `prc.tail_model` head, so the
desk's assignment points Q09 at **`L2-hub8725`**, the retro-frame-v2 LightGBM member of
stack-5 (weight 0.154 in blend-5's anchor), which carries no stand-chain column at all.
I take that reading and add three decisions of my own.

**1. Three arms, not two, and the third one is the assignment's own repair moved to the
front.** `none` is the shipped v2 frame; `chain` adds `ts_gap_arr1..3` exactly as
`prc/tail_features.py` computes them and as the retro **v4** frame imports them; `derived`
adds those three plus exp-037's `slack`, `slack2`, `log_g`, `n_dep_between`,
`arr_stand_per_day`. The assignment names seven exp-037 columns and two of them —
`g_any`, `g_prev2` — are the same numbers as `ts_gap_arr1`, `ts_gap_arr2`. exp-037 §0
measured that on the noac rows; `code/equality.py` measures it here on **every departure
of every month**, which is the population Q09 is scored on. Two bit-identical float
columns in front of a tree that samples 80 % of its features is a change to the sampling
and not a change to the information, so the `derived` arm carries them once. Running the
repair arm in the same session as the pair costs one more fit and removes a whole round
trip if the first pair is flat.

**2. The reproduction gate is against exp-025's Kaggle baselines, not the published OOF.**
exp-025 §10.6 measured that a shift taken against the workstation's published member OOF
is mostly the machine: `L2-hub8725`'s machine term is +0.096 / −0.003 / +0.471 s by fold
and changes sign. The same-environment baselines are
`experiments/exp-025-cloud/oof-nested/L2-hub8725--{fold}.parquet`
(314.5038 / 227.1649 / 350.8636). The `none` arm must land on those, and the kernel
refuses to report a `chain` number if it does not.

**3. The chain is priced twice, against two baselines, and both are reported.** The
contract's endpoint is paired all-row MSE against **v10**; that comparison also carries
the machine term, because the arm's member was fitted on Kaggle and v10's was not. So
every arm is also priced against the **`none` arm's own serving vector**, which is the
same member producer in the same session with only the column list changed. The first
number decides whether anything ships; the second is the one that answers "does the chain
carry inside this member", and a candidate needs both. `code/q09.py` gates its serving
path on exp-038's gate F first — v10 rebuilt from the committed member frames to 0.0000
on every fold — so the path is exp-038's before either baseline is read.

## Readings rejected

* **The tail member (`T-tailv4`) or `prc.tail_model`.** exp-037 §7: the chain is already
  in every tail head, so a chain arm there is a no-op. Rejected by measurement, not taste.
* **`prc/nonm.py`.** It carries no chain either, and it is a published head — but it is
  not one of stack-5's seven members and its rows are 1.4 % of the file, so it cannot
  produce the final-predictor benefit Q09's advance clause requires.
* **A tail-head refit inside the stack** (exp-037 §10 item 3). It needs the tail member's
  own rebuild and cannot be resolved on 491 noac rows (§10 item 4).
* **Scoring this on the noac rows.** The chain columns move every row's features in
  `L2-hub8725`, so the paired gain is read on all rows at the contract's endpoints. The
  388 ranking noac rows are reported as a served shift, and they are held bit-identical
  to v10 by the recipe's own noac hold, which is checked rather than assumed.
* **A laptop-side three-fold fit.** `assignments/LAPTOP.md` sends it to Kaggle, and the
  same-environment gate needs Kaggle anyway.

## What runs where

| step | where | cost |
|---|---|---|
| the chain columns, and their equality with exp-037's | laptop, 4 threads | minutes |
| 3 arms x 3 folds, the member refit | Kaggle CPU, one kernel per fold | ~2.7 h a fold, 0.0 GPU-h |
| 3 arms, the twelve-month ranking member | Kaggle CPU, one kernel | ~1 h, 0.0 GPU-h |
| the v10 serving path and the contract pricing | laptop, 4 threads | ~20 min an arm |

`C2-cat` (Kaggle GPU, `PRC_GPU=0`, cap 2 GPU-h) runs only if `L2-hub8725` gives `A > 0` on
all three folds.
