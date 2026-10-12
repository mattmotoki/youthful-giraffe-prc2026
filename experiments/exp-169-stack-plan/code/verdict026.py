"""Explicit Markdown review verdicts; examples cannot authorize a build."""
import re
from pathlib import Path

_LABEL = re.compile(
    r'^(?:(?:(?:overall|final)\s+){0,2}verdict|fresh-context correctness review)\b',
    re.I,
)
_STATUS = re.compile(
    r'^[\s\[(]*(PASS(?:ED)?|FAIL(?:ED|URE)?|BLOCK(?:ED)?|REJECT(?:ED)?)\b',
    re.I,
)


def parse(text):
    """Read all declarations, ignoring quoted, fenced and indented examples.

    Unknown explicit statuses reject approval too. A heading with no status is
    not a declaration. Tables can put the label and status in separate cells.
    """
    out = []
    fence = None
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if fence:
            if re.fullmatch(re.escape(fence[0]) + '{' + str(fence[1]) + r',}\s*', line):
                fence = None
            continue
        if line.startswith('>'):
            continue
        marker = re.match(r'^(`{3,}|~{3,})', line)
        if marker:
            run = marker.group(1)
            fence = (run[0], len(run))
            continue
        if raw.startswith(('    ', '\t')):
            continue
        plain = line.replace('**', '').replace('__', '').replace('`', '')
        plain = re.sub(r'^#{1,6}\s*', '', plain)
        plain = re.sub(r'^(?:[-*+]\s+|\d+[.)]\s+)', '', plain).strip()
        cells = [cell.strip() for cell in plain.strip('|').split('|')]
        for index, cell in enumerate(cells):
            label = _LABEL.match(cell)
            if not label:
                continue
            remainder = cell[label.end():]
            explicit = bool(re.match(r'^\s*[:=\-–—]', remainder)) or (
                not remainder.strip() and index + 1 < len(cells)
            )
            tail = ' '.join([remainder, *cells[index + 1:]])
            tail = re.sub(r'^[\s:=\-–—]+', '', tail)
            if not tail:
                continue
            status = _STATUS.match(tail)
            if not status and not explicit:
                continue  # Prose such as "Verdict parser gaps" is not a verdict.
            verdict = 'UNKNOWN'
            if status:
                word = status.group(1).upper()
                verdict = next(root for root in ('PASS', 'FAIL', 'BLOCK', 'REJECT') if word.startswith(root))
            out.append({'line': number, 'verdict': verdict, 'text': line})
    return out


def require_pass(path):
    out = parse(Path(path).read_text())
    assert out and all(r['verdict'] == 'PASS' for r in out), f'Explicit unopposed PASS required: {path}: {out}'
    return out
