# Reference material

This directory keeps source material that can shorten the path to a strong PRC
2026 baseline without confusing prior work with the new solution.

## Prior local work

The local snapshot of an earlier competition is private process material.
It is excluded from this source export and is not an input to the model.

## External code to study, not copy

- [NASA ML-airport-taxi-out](https://github.com/nasa/ML-airport-taxi-out), pinned
  for review at `add029a55966d846498d42ac54d698d3ed724775`. It separates
  unimpeded and congestion-driven taxi-out time and uses stand, runway, carrier,
  surface counts, and XGBoost models.
- [NASA ML-airport-data-services](https://github.com/nasa/ML-airport-data-services),
  pinned for review at `3389776d67e8d2564b627434296c9b62b82ec380`.
  It contains surface-count and airport-configuration utilities used by the
  taxi-out project.

These NASA repositories use the NASA Open Source Agreement 1.3 for relevant
code. The PRC prize rules require produced source code to be published under
GNU GPLv3. Until license compatibility is resolved, use the NASA projects to
learn mechanisms and reimplement those mechanisms from the published ideas;
do not copy their source into the competition solution.

## Prior PRC challenges

- [PRC 2024 challenge](https://ansperformance.eu/study/data-challenge/dc2024/)
  and [public team repositories](https://github.com/prc-data-challenge-2024)
- [PRC 2025 challenge](https://ansperformance.eu/study/data-challenge/dc2025/)
  and [public team repositories](https://github.com/prc-data-challenge-2025)

These editions have different targets. They remain useful for OpenSky data
handling, submission code, reproducibility, validation discipline, and the
standard expected for a prize-eligible public repository.

The fuller source-by-source review and literature shortlist is in
[`docs/reference-research.md`](../docs/reference-research.md).

## Transfer rule

For each adopted mechanism, record:

1. its source;
2. the PRC 2026 column and timestamp that implement it;
3. why the value is available when a ranking prediction is made;
4. the validation split on which it will be tested; and
5. the measured change in RMSE.

This rule is especially important because the ranking data blank departure
off-block time and taxi-out time but retain other actual, scheduled, movement,
runway, stand, and flight-plan fields.
