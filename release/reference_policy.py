"""Keep public narrative references inside the public source boundary.

Executable identifiers, model names, stand IDs and encoded arrays are data,
not desk decision references. Python comments and docstrings are narrative;
other strings remain executable contracts. Markdown table cells remain data.
Explicit references to the private decision file are forbidden in every format.
"""
from __future__ import annotations

import ast
import base64
import gzip
import csv
import zlib
import io
import json
import posixpath
import re
import tokenize
from urllib.parse import unquote, urlsplit

from release_set import is_released
from label_glossary import glossary_lines

DECISION = re.compile(r"(?<![\w+-])D[1-9][0-9]{0,3}(?![\w-])")
DECISION_FILE = re.compile(r"(?i)(?<!\w)(?:docs/)?DECISIONS\.md(?!\w)")
LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(?:<([^>\n]+)>|([^\s)]+))[^)\n]*\)")
REFERENCE = re.compile(r"^\s{0,3}\[[^\]\n]+\]:\s*(?:<([^>\n]+)>|([^\s]+))", re.M)
HTML_LINK = re.compile(r'''\b(?:href|src)\s*=\s*["']([^"']+)["']''', re.I)


def local_target(rel: str, target: str) -> str | None:
    url = urlsplit(target)
    if url.scheme or url.netloc or not url.path:
        return None
    path = unquote(url.path).replace('\\', '/')
    resolved = posixpath.normpath(path.lstrip('/') if path.startswith('/') else
                                 posixpath.join(posixpath.dirname(rel), path))
    return resolved


def narrative(text: str, rel: str) -> list[str]:
    if rel.endswith('.py'):
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        parts = [tok.string for tok in tokens if tok.type == tokenize.COMMENT]
        try:
            tree = ast.parse(text)
        except SyntaxError:
            # Some historical sources need Python 3.12 f-string grammar.
            # Their comments and triple-quoted narrative still get checked.
            parts.extend(tok.string for tok in tokens if tok.type == tokenize.STRING
                         and re.match(r'''(?i)^[rubf]*(?:"""|'{3})''', tok.string))
            return parts
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    parts.append(doc)
        return parts
    if rel.endswith(('.md', '.txt', '.rst')):
        # Only an explicitly labelled stand column is a data-ID exception.
        parts, stand_columns = [], set()
        for line in text.splitlines():
            # Public rule definitions are not citations of private records.
            if rel == 'README.md' and line in glossary_lines():
                continue
            if not line.lstrip().startswith('|'):
                stand_columns = set()
                parts.append(line)
                continue
            cells = line.split('|')
            labelled = {i for i, cell in enumerate(cells)
                        if cell.strip().lower() in ('stand', 'stand_mvt', 'stand id')}
            if labelled:
                stand_columns = labelled
            parts.extend(cell for i, cell in enumerate(cells)
                         if not (i in stand_columns and DECISION.fullmatch(cell.strip())))
        return parts
    if rel.endswith('.sh'):
        return [line for line in text.splitlines() if line.lstrip().startswith('#')]
    if rel.endswith('.json'):
        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for key, item in value.items():
                    if key == 'data' and value.get('encoding') == 'gzip-base64':
                        try:
                            raw = gzip.decompress(base64.b64decode(item, validate=True))
                        except (OSError, EOFError, ValueError, zlib.error) as error:
                            raise ValueError('invalid encoded model array') from error
                        if value.get('dtype') != '<f8' or not isinstance(value.get('rows'), int) or len(raw) != 8 * value['rows']:
                            raise ValueError('encoded model array shape mismatch')
                        continue
                    yield from strings(item)
            elif isinstance(value, list):
                for item in value:
                    yield from strings(item)
        return list(strings(json.loads(text)))
    if rel.endswith(('.csv', '.tsv')):
        reader = csv.DictReader(io.StringIO(text), delimiter='\t' if rel.endswith('.tsv') else ',')
        return [value for row in reader for key, value in row.items()
                if isinstance(value, str) and (key or '').lower() not in ('stand', 'stand_mvt', 'stand id', 'stand_name_mvt', 'xplane_stand_name')]
    if rel.endswith('.ipynb'):
        notebook = json.loads(text)
        return [''.join(cell.get('source', [])) for cell in notebook.get('cells', [])
                if cell.get('cell_type') == 'markdown']
    return [text]


def scan_references(text: str, rel: str) -> dict[str, list[str]]:
    prose = narrative(text, rel)
    link_text = '\n'.join(prose) if rel.endswith(('.py', '.sh', '.json')) else text
    if rel.endswith('.py'):
        # String constants can render links; code brackets are not Markdown.
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.STRING:
                try:
                    value = ast.literal_eval(token.string)
                except (SyntaxError, ValueError):
                    continue
                if isinstance(value, str):
                    link_text += '\n' + value
    targets = [a or b for pattern in (LINK, REFERENCE) for a, b in pattern.findall(link_text)]
    targets.extend(HTML_LINK.findall(link_text))
    excluded = set()
    for target in targets:
        path = local_target(rel, target)
        if path is None:
            continue
        # Old notes sometimes write repository-root paths without a leading slash.
        # Check that spelling as well as the ordinary relative resolution.
        raw = posixpath.normpath(unquote(urlsplit(target).path).replace('\\', '/').lstrip('/'))
        candidates = {path, raw} if raw.startswith(('docs/', 'release/', 'prc/', 'experiments/', '.desk/')) else {path}
        for candidate in candidates:
            probe = candidate + '/placeholder.py' if '.' not in posixpath.basename(candidate) else candidate
            if candidate == '..' or candidate.startswith('../') or not is_released(probe):
                excluded.add(candidate)
    decisions = {m.group() for part in prose for m in DECISION.finditer(part)}
    if any(DECISION_FILE.search(part) for part in prose):
        decisions.add('private decision file')
    return {'excluded_links': sorted(excluded), 'decision_references': sorted(decisions)}
