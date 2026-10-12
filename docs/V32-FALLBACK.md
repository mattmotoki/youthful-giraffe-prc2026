# Taxi-out time: the accepted entry v32

Team youthful-giraffe, PRC Data Challenge 2026. This document describes the older v32 fallback kept by v35.
The scorer returned **242.7480 s RMSE** for **344,841 departures**. The file
has **4,707,529 bytes** and SHA-256
`f09f609429c81fd888e19e75e16f51b57978d55cc2639d6161c894da0baaa0de`.
The committed public records are [PUBLIC-MODEL-v32.json](PUBLIC-MODEL-v32.json),
[PUBLIC-SUBMISSIONS.md](PUBLIC-SUBMISSIONS.md), and
[the replay contract](../release/V32-REPLAY.json).
The fallback source and replay hashes are in the records linked above.

## 1. Task and inputs

The target is take-off time minus airport off-block time, in seconds. The
ranking data retains the departure's actual take-off and Network Manager
actual off-block fields. The two departure target fields are blank. The task
is a reconstruction after operations. We use the supplied fields and open
external observations. See [input-rule.md](input-rule.md) and
[external-data-rules.md](external-data-rules.md).

Open adsb.lol ground tracks supply an observed off-block clock and its quality.
The earlier surface rule uses the first lasting displacement from the stand.
The v23 stage adds an implied off-block clock from these observations, stand
geometry and row timing. An annual learner adds airport surface-state proxies.
Rows without an aircraft type retain the older base prediction.

## 2. Model composition

| Entry | Update | Rows moved | Floor | RMSE s |
|---|---|---:|---:|---:|
| v22 | Older blend, residual stack, surface heads and queue correction | Base | 60 s | 245.0207 |
| v23 | Implied-clock residual update, with LIRF held at v22 | 186,168 | 60 s on moved rows | 244.1236 |
| v24 | Annual no-clock learner with surface-state proxies | 148,197 | 60 s on moved rows | 242.8064 |
| v32 | Annual proxy learner on supported with-clock rows | 173,539 | 60 s on all rows | 242.7480 |

The table's counts, floors and upload hashes are in
[the replay contract](../release/V32-REPLAY.json). The v23 raw vector holds
5,571 LIRF rows. One moved v23 row and 30 moved v24 rows are clipped at the floor.
No v32 row is clipped. The stages do not change any row without an aircraft type.

The v24 weight is **0.4967252524893474**, fitted on 2025 calibration folds.
The v32 fold weight is **0.12797310003895088**; the final kernel serves
that same full-precision weight. These values come from 2025 labels, as recorded in
[PUBLIC-MODEL-v32.json](PUBLIC-MODEL-v32.json). The v32 final fit has **2,083,537 rows**,
**172 features** and **1,500 rounds**. The exp-136 code supplies inputs for
v32; its wider-window fitted model is not part of the entry.

The fitting sources are exp-127, exp-130, exp-134 and exp-137. Their source
pins and the exp-136 input pin are in PUBLIC-MODEL-v32.json. Export changes set
caller-owned paths and account names and remove private packaging. They do
not change the fitted model rules. [REPRODUCE.md](REPRODUCE.md) gives the path
from raw data to each derived input package.

## 3. Validation and board history

The weights were fitted on the old 2025 month folds. Those folds do not give
an isolated estimate of the served model. Earlier fitted base predictions
carry held-month labels into later stages. V24 also uses different fold and
serving bases. V32 inherits both paths. We preserve the accepted prediction
and state these faults; this release does not repair or re-score the model.
The full fault list is in PUBLIC-MODEL-v32.json. The later validation protocol is
[VALIDATION.md](VALIDATION.md); it does not validate the old vectors again.

Uploads **v26-v29, v31 and v33** were probes made only to read
hidden score responses. They remain in [PUBLIC-SUBMISSIONS.md](PUBLIC-SUBMISSIONS.md)
as history. The current rule stops that use. **No released parameter, residual, mask or
rate comes from a board reading.** V32 uses its recorded 2025-fitted update.
Board results check the accepted file's score; they do not prove transfer to
new dates or airports.

The exp-137 stage failed its frozen gain gate. The final proxy fit proceeded
under a positive-point rule: a positive gain estimate on the supported
with-clock cells allowed an all-year fit and an update on those cells. Passing
the frozen gain gate was not required under that rule. The gate failure and
the recorded fitted weight remain part of the model history.

## 4. External data and licences

The source is GNU GPL version 3; see [LICENSE](../LICENSE).

- adsb.lol globe_history archives: Open Database Licence (ODbL-1.0), with
  contributed contents under CC0-1.0. Data © adsb.lol contributors.
- OurAirports airport and runway geometry: public domain.
- X-Plane Scenery Gateway and OpenStreetMap stand geometry: separate upstream
  GPLv2 and ODbL terms. See [GEOMETRY-DATA.md](GEOMETRY-DATA.md); the old
  Gateway or-later tag is not verified by its primary upload terms.
- Copernicus ERA5: CC BY 4.0. The exact derived table is a separate data bundle
  under that licence. It retains its original bytes and absent version attr.

Contains modified Copernicus Climate Change Service information 2026.
Neither the European Commission nor ECMWF is responsible for any use of this
information. The two ERA5 product DOIs, retrieval dates and transformations
are in [weather-provenance.md](weather-provenance.md). The archive and match
rules are in [EXTERNAL-DATA.md](EXTERNAL-DATA.md).

The organizer's recorded post-operations answer permits open observations of
the ranking months. This removes the earlier assumed event-time cut-off.
It does not certify this entry's eligibility. Open licences, public access and
reproduction instructions remain required. Private Kaggle packages contain
challenge-derived inputs; they are not separate open external sources.

## 5. Reproduction limits

The exported source, saved prediction arrays and official ranking/template
files replay v23, v24 and v32 exactly. The arrays have no movement IDs or
training labels. Replay checks input hashes, saved-array hashes, output-value
hashes, output-file hashes and stage counts. It fits nothing.

**No fresh fit has reproduced the saved vectors.** The seven original private
Kaggle input package versions were checked: 67 recorded files matched; three
more had no recorded hash. The original preparation sources exist, but a
reproducer must rebuild the derived packages from challenge data and open
sources. Exact refit is unproved. The measured stage runs imply about **21
Kaggle CPU hours** after inputs exist; the v22 base rebuild adds further work.
Clean-score outputs can differ because private row exclusions are withheld.

The original v23 fitter writes only the last fold vector. The repaired fold
file used by v32 was rebuilt from its full prediction output. This is a
required step in a new fit. Other label, scope and floor faults remain in the
public fault record. [REPRODUCE.md](REPRODUCE.md) states what must be rebuilt
and which checks have not run. The release manifest states the exported paths
and their hashes. Nothing in this release is a claim that
a full rebuild or public publication has already succeeded.
