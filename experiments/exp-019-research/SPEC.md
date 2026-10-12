# exp-019 research — SPEC

## My reading of the assignment, and the readings I rejected

**The reading I took.** The deliverable is a *decision document*, not a
literature review: a ranked list of at most eight local experiments, each with a
cost and a gate, each mapped to the ledger row that already measured it or
explicitly marked unmeasured. Sources matter only insofar as they change what we
would run. So the sweep is organised around one question — what is
`AOBT_3_flt`, mechanically — because that is the object every model in this repo
spends its capacity on and the one the ledger describes without explaining.

**Readings I rejected.**

1. *"Survey the taxi-out prediction literature."* Rejected: the ledger already
   measured the literature's central claim (take-off queue size) with a ceiling
   — `exp-004 oracle` shows a **perfect** queue feature reaching only 269.3 —
   so a survey would return proposals with receipts against them. The sweep
   spends its budget on the mechanism nobody here has named instead.
2. *"Treat the EUROCONTROL specification as ground truth."* Rejected, and the
   protocol says why: a response is a proposal, not an authority. The DPI guide
   says what implementers are told to do, not what any airport's system did in
   2025. Its sharpest local prediction was tested and **failed** in its strong
   form, which is recorded and which is why E1 exists.
3. *"Propose the experiments the sources suggest."* Rejected where a receipt
   exists. Every mechanism class in the desk's question list is mapped to its
   controlling measurement first; three of the eight proposals are consequently
   framed as *tests of why* an already-measured effect works, not as new gains.
4. *"Send the organizer question."* Rejected as outside a worker's authority:
   it is outward-facing, `internal decision record` already carries an organizer
   question as supplied-input rule, and E8 is written as a rider for the desk to send.

## Frozen before any source was read

- A family is excluded only with a **receipt** — a named ledger row and its
  controlling number. An unmeasured neighbour is written "disfavoured, with the
  exception that…", never banned. (`EXTERNAL_RESEARCH_PROTOCOL.md` cites a case
  where an unmeasured ban hid a real 31 % improvement.)
- No fits. Local checks are permitted where they cost seconds and can falsify a
  source's claim; three were run (clock resolution, whole-minute `MVT − AOBT`,
  the 5,400 s ceiling) and all three are reported including the one that
  contradicted the source.
- At most eight experiments, ranked by expected board seconds per 4-thread hour.

## Compute

4 threads (`docs/COMPUTE.md` compute allocation), no fits, no GPU. The only local work was three
`prc.data` reads.
