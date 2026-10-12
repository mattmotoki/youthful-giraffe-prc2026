"""Public meanings of opaque rule, diagnostic and test labels."""

LABEL_RULES = {
    'D1': 'Diagnostic section: reconcile the no-NM row count; this is not a model rule.',
    'D2': 'In exp-169 assembly records, the mean squared difference between two prediction vectors, in s² per row; this is not a rule.',
    'D3': 'In exp-014, report airport quantile shares; in exp-012, follow the team compute split with lane thread caps and GPU allocation; lowercase d3_ weather fields are three-hour differences.',
    'D6': 'Early release selection: serve the recorded blend-config-5 as entry v2.',
    'D9': 'Nested validation rule: exclude the scored month from member fits used to estimate blend weights and stackers.',
    'D10': 'Nested validation work: rebuild member predictions with the scored month excluded before comparing model changes.',
    'D11': 'No board fitting: submit model candidates, never probes designed to recover hidden labels.',
    'D18': 'In fitting code, alias of the exp-018 dataset module; also a test example of an opaque source identifier. It sets no fitted model rule.',
    'D20': 'Tail verdict: keep the tail loss visible as a secondary guard, with the historical pooled reading stated.',
    'D23': 'Discounted gain gate: cap credited fold gain at its registered prediction and require each credited interval to clear zero.',
    'D24': 'Seasonal head rule: choose a separate fitting panel for each served month by cyclic month distance.',
    'D28': 'Redundancy test: compare added-month fits with matched draws before making a new serving claim.',
    'D32': 'Same-season transport rule: the tested carrier path is closed provisionally; the destination remains open to another mechanism.',
    'D36': 'Diagnostic advance rule: no negative fold point and clear intervals on two folds, with matched raw-key controls.',
    'D43': 'Origin diagnostic: use a frozen block-value ladder and matching band; do not treat it as a causal season test.',
    'D46': 'Near-origin diagnostic: choose the closest qualifying frozen row, stop if none matches, and create no candidate.',
    'D47': 'Sparse diagnostic contract: keep the registered split, controls, order and stopping rules fixed; no serving claim follows.',
    'D107': 'Frozen gain gate: require positive fold gains, sham and placebo controls, and clear pooled and per-fold intervals before the registered advance.',
    'D119': 'Test example of a forbidden narrative citation; it sets no fitted model rule.',
    'D123': 'Candidate review rule: a fresh-context correctness review of each exact candidate file before any upload.',
    'D132': 'Positive-point rule: a positive supported-cell gain estimate allowed the final proxy fit even when the frozen gain gate failed.',
    'D134': 'Current score-feedback rule: no fitted weight, residual, mask or rate comes from a board reading.',
}


def glossary_lines():
    return [f'- `{label}` — {rule}' for label, rule in LABEL_RULES.items()]
