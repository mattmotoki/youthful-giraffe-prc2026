"""Check inline repository-path citations against the actual export membership."""
from pathlib import PurePosixPath
import posixpath
import re
from urllib.parse import unquote

ROOTS = ('docs/', 'prc/', 'scripts/', 'release/', 'references/', 'experiments/',
         'assignments/', '.desk/', 'data/', 'artifacts/', 'submissions/',
         'modeling/', 'notebooks/', 'catboost_info/')
INLINE = re.compile(r'(?<!`)`([^`\n]+)`(?!`)')


def is_top_document(rel):
    path = PurePosixPath(rel)
    return rel == 'README.md' or path.parent == PurePosixPath('docs') and path.suffix == '.md'


def missing_path_citations(text, rel, exported_paths):
    if not is_top_document(rel):
        return []
    paths = set(exported_paths)
    issues, fence = [], None
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        opening = re.match(r'(`{3,}|~{3,})', stripped)
        if opening:
            mark = opening.group(1)[0]
            if fence is None:
                fence = mark
            elif mark == fence:
                fence = None
            continue
        if fence is not None:
            continue  # Commands and caller-owned input/output layouts are instructions.
        for value in INLINE.findall(line):
            token = unquote(value.strip()).split('#', 1)[0]
            if token == 'assignments/...':
                continue  # Naming convention explained by the required roles paragraph.
            if re.search(r'\s', token) or '://' in token or not token:
                continue
            rooted = token.lstrip('/').startswith(ROOTS)
            relative = token.startswith(('./', '../'))
            local_md = '/' not in token and token.endswith('.md')
            root_file = token in ('LICENSE', 'README.md', 'requirements.txt')
            if not (rooted or relative or local_md or root_file):
                continue
            target = posixpath.normpath(token.lstrip('/') if rooted or root_file else
                                       posixpath.join(posixpath.dirname(rel), token))
            present = target in paths or any(p.startswith(target.rstrip('/') + '/') for p in paths)
            if not present:
                issues.append({'line': number, 'citation': value, 'target': target})
    return issues
