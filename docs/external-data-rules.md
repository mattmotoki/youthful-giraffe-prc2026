# External data rule for v32

The [eligibility page](https://prc-data-challenge-2026.netlify.app/eligibility.html)
requires open, documented external data, an open licence for added data,
GPLv3 produced code on GitHub and enough instructions to reproduce the result.
The official [data page](https://prc-data-challenge-2026.netlify.app/data.html)
and [ranking page](https://prc-data-challenge-2026.netlify.app/ranking.html)
define the supplied inputs and score population.

The organizer's recorded September answers state that the task is after
operations, that the supplied fields are valid inputs, and that open ranking-
month trajectories, including own-flight ground positions, can be used.
They remove the earlier assumed event-time cut-off. The private conversation
record is withheld. This is a summary of that answer, not approval of this
specific entry. OpenSky state vectors are not an approved external source;
no such source is in v32.

The current rule prohibits probes that solve for hidden answers from score responses.
Uploads v26-v29, v31 and v33 are history. No released parameter, residual,
mask or rate uses those readings. Board checks can test local-to-board
alignment; they are not training labels.

V32 uses adsb.lol, ERA5, OurAirports and Gateway/OSM geometry. Their sources,
terms, frozen hashes, transformations and reproduction routes are in
[EXTERNAL-DATA.md](EXTERNAL-DATA.md),
[weather-provenance.md](weather-provenance.md),
[GEOMETRY-DATA.md](GEOMETRY-DATA.md) and [REPRODUCE.md](REPRODUCE.md).
The original private Kaggle datasets contain derived challenge inputs. They
must be rebuilt; their private status is not evidence that the underlying
external sources are closed. A fresh fit is still unproved.
