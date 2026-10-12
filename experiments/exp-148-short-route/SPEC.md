# Public specification of accepted v35: R2b-w100-kno

This is the release specification of the recorded exp-148 method. The fitting
sources keep the split checks, member settings, feature list, model parameters,
CV and serving formulas of the accepted run. Original working specifications
and transport records are not distributed. The source manifest pins this copy.
No new fit has reproduced its predictions.

## 1. Members and split isolation

Eight members: T-tailv4, C2-cat, L2-hub8725, D1-winter, F1-fresh, N1-nmtail,
L and Lp. The first five form the row-mean anchor. The six base-member drivers
are under `../exp-147-foldserve/code/members/`; L and Lp use `learner_wp.py`
in that experiment. Both fold and ranking predictions use these same drivers.
They keep the published member model rules while changing training-day masks
and preventing scored-month labels from entering scored-row encodings.

The `janjul` fits exclude every day of January and July 2025 plus their month
buffer days. Each `jjx-MM` fit, for MM in 12, 02, 06, 08, also excludes every
day of its scored month and its buffers. Its held rows train the stacker.
Final members fit all 2025 labels and score the two ranking months.
Every member artifact is written and read through `pilot.write/load`, with
split ownership, the admission rule, the split fingerprint and a byte hash.

## 2. Level 2

Owner r01 trains on December, February, June, July and August, with January
and its buffers excluded. Owner r07 substitutes January for July. Owner
rfinal trains on all six months and scores ranking rows from the final members.
The recorded final training frame has 1,028,150 rows. The two boundary rows
without features are dropped only when their IDs match the caller-derived
exception file; any other missing row stops the fit. The public source holds
no IDs. `scripts/exp148_missing_features.py` derives the two exceptions from
the following month's challenge files, and PRC_R2_NO_FEATURES selects that file.

The target is y minus the row mean of the first five members. Inputs are
member deviations, the anchor, row context, observed surface and shipped
features, label-free queue columns, same-day queue residuals against this
anchor, file and observed taxi, proxies, two row clocks and five differences.
R2 removes every v23-prefixed feature and ctx_month. `feature_list` is the
exact ordered feature-list rule. Airport and class are categorical.

LightGBM and CatBoost fit residuals. Their exact parameter dictionaries are in
`code/stacker.py`. Each chooses rounds with leave-one-training-month-out CV,
maximum 5,000 rounds, then refits on all training rows at the rounded mean of
its best CV rounds. Recorded rfinal rounds are 461 and 2,208 respectively.
The chosen form is max(anchor + (raw_lgb_residual + raw_cat_residual)/2, 60 s).
The two predictions are averaged before the floor.

## 3. Variant and serving

The variant grid was fixed before R2 scores and upload: weights 1, 0.75, 0.5,
0.25 on the chosen form (the balance on the served anchor), crossed with keeps
of none, no-aircraft-type, ordinary without AOBT, or both. The highest joined
January/July gain against the old v32 reference chose weight 1 and both keeps.
Ties below 1 s² per ranking row favour higher weight and the smaller keep set.
The mean form won the recorded training-month CV form comparison.

The published choice is in `release/V35-CHOICE.json`. v32's served values replace
the new prediction in unsupported cells and the chosen keep sets. Unsupported
means the frozen cell inventory marks the airport/month/class cell unsupported.
The raw AOBT field and the rebuilt class determine the ordinary-without-AOBT set.
The recorded counts are 388 noac, 3,205 ordinary without AOBT, 22,721 unsupported,
26,310 in their union. Another value is equal by chance: 26,311 unchanged rows.
Ranking output has 344,841 unique template IDs, finite predictions and a 60 s floor.
`code/serve_r2.py` implements the complete rule.

## 4. Recorded evidence and limits

The chosen local gain was 178.2 s² per ranking row, 95% interval [-365.8, 688.3],
P(G <= 0) 0.245. January contributes -40.6, July 218.8. R2b without keeps gives
-27.9. The keep sets account for about 206; their local v32 references inherit
held-label paths, so they are optimistic. The accepted board score is 241.7342.
No weight was fitted to that score. Keep the published choice when refitting
v35; a new selection is a new candidate.

A fresh fit needs the original challenge and open-source versions, all
recreated monthly caches, classes and member inputs, and the rebuilt R2 tables.
It also needs the exact v32 fallback. A replay of saved values proves bytes,
not a new fit. The historical reference generator `code/refs.py` reads saved
old month-fold artifacts from private history; those artifacts do not ship.
Recomputing the original variant selection therefore needs those original
reference vectors or new fold outputs, and can give another choice.

## 5. Fixed blend for v36

B50 is 0.5 × v35 + 0.5 × v32 in official template order. The weight was fixed
before its score. `code/serve_blend.py` aligns the two vectors, checks unique
population, finite values and a 60-second floor, then writes the mean. V36's
accepted hash and score are in `docs/PUBLIC-MODEL.json`. V35 and v32 remain
separate replay targets and separate fresh-fit routes. Partner selection used
an entry comparison, including v34's board loss; the weight was not fitted.
