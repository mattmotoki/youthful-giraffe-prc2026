# Stand geometry used by v23, v24 and v32

The stand-to-runway lookup is
`experiments/exp-127-clock-precision/stand_geometry.csv`. It contains airport
and stand geometry, source URLs, scenery or OSM IDs, source hashes and match
rules. It has no movement IDs or taxi-time labels. Its source licences are
separate from the repository's GPLv3 code licence.

Gateway stand and runway geometry comes from X-Plane Scenery Gateway airport
scenery (`apt.dat` stand and runway records). The per-row scenery URL and hash
identify the open source used. [Gateway upload terms](https://gateway.x-plane.com/docs/TermsAndConditionsForUploadOfAirportScenery.pdf)
state GNU GPL version 2 without an explicit later-version choice. The README
inside the [scenery 105504 pack](https://gateway.x-plane.com/apiv1/scenery/105504)
(LTFM) separately offers version 2 or any later version. These are two primary
source notices; the pack README gives a choice that the upload terms do not state.

The Gateway-derived portion retains the GPLv2 terms. The full licence text is
[data-licences/GPL-2.0.txt](data-licences/GPL-2.0.txt). The old CSV tags say
`GPL-2.0-or-later` and remain historical metadata. The one checked pack does not
establish the same later-version choice for every other source pack. Check each
recorded source pack before claiming a broader grant. No Gateway code is linked
into the model.
The exact source scenery must remain obtainable at its recorded URL or as a
separate upstream-licensed source copy before public data distribution.

OpenStreetMap fallback positions are © OpenStreetMap contributors, under the
[Open Database Licence 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
See [OpenStreetMap copyright](https://www.openstreetmap.org/copyright).
The OSM-derived portion retains ODbL terms, including attribution and the
same licence for a published derivative database. The table records its OSM
source IDs and tags. Do not relicense the table as GPLv3 code.
The refit sources (`refit-src/`) also ship the exp-127 file
`osm_parking_positions.csv`, which is OpenStreetMap data under ODbL 1.0 with
the same attribution, and the final stand table `stand_geometry-final.csv`,
which has the same mixed terms as this table.

The source preparation is exp-127 `build_stand_geometry.py`,
`fetch_osm_parking_positions.py` and `merge_osm_geometry.py`. Gateway matches
have priority; OSM adds missing stand names. The lookup copies coordinates,
normalizes stand names and calculates stand-to-runway distances. Original
recorded row licence tags are retained as historical metadata, with the
correction above. Source hashes fix the snapshots; a current fetch can differ.

The code and data are separate works in this export. No claim is made that
all mixed-table licence obligations have been independently reviewed. The
final release review must check the notices and source availability. Refit
and replay do not by themselves certify prize eligibility.
