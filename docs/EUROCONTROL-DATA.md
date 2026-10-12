# EUROCONTROL monthly taxi-out source

The v37 and v40 corrections uses monthly taxi-out means from EUROCONTROL's
[Taxi-Out Additional Time workbook](https://www.eurocontrol.int/performance/data/download/xls/Taxi-Out_Additional_Time.xlsx).
Source attribution: © European Organisation for the Safety of Air Navigation (EUROCONTROL).
The [indicator definition](https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf)
describes taxi-out time and its additional-time measure.

The source was read on 2026-09-30 at 06:33:29 UTC. The unchanged workbook was
1,371,583 bytes, SHA-256
`82d119ed3cfaf7a3c2f29b27ce942f0a36ea599b3af76e777e4eb354a7a4adb8`.
Its DATA sheet has 240 airport-month rows for 2025 and 2026 at the ten challenge
airports. Source means in minutes are converted to seconds for the model.
The workbook and source rows are not included in this export.

The [EUROCONTROL copyright terms](https://ansperformance.eu/about/disclaimer/)
require source attribution and permit copying for non-commercial purposes
without financial gain. Modification of source information requires prior
written permission. EUROCONTROL gives no warranty; views do not necessarily
represent its official position. These terms govern the source information,
separately from the GNU GPL version 3 licence for this repository's code.

The [frozen parameter table](../experiments/exp-151-published-means/frozen-parameters.json)
is the team's derived model record. Coverage coefficients, corrected targets
and shrinkage weights are model calculations; they are not an official
EUROCONTROL product or a revised source workbook. The public table also records
the published mean used in each calculation. The source attribution and terms
above apply to these source-derived values. The [extractor](../experiments/exp-151-published-means/code/extract_source.py)
reads a caller-supplied workbook; the [fit source](../experiments/exp-151-published-means/code/step_d.py)
records the calculation. A new source download can differ from the pinned file.

The copied model table contains all 40 published_mean_s values used by its
development and ranking calculations; this attribution and the source terms
apply to both sets. The original results table and extracted source tables
are excluded. V40 reuses these exact parameters without a new fit.
