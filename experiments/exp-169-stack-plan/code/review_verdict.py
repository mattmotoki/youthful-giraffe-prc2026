"""Accept explicit verdict declarations; quoted PASS text is not a verdict."""
import re
from pathlib import Path


def verdict_lines(text):
    found=[];fenced=False
    for number,line in enumerate(text.splitlines(),1):
        line=line.strip()
        if line.startswith(('```','~~~')):
            fenced=not fenced
            continue
        if fenced or line.startswith('>'):continue
        plain=line.replace('**','')
        match=re.match(r'^(?:VERDICT|Verdict|Fresh-context correctness review):\s*(PASS|FAIL|BLOCK)(?=$|[\s.,;])',plain)
        if match:found.append({'line':number,'verdict':match.group(1),'text':line})
    return found


def require_pass(path):
    lines=verdict_lines(Path(path).read_text())
    assert lines and {r['verdict'] for r in lines}=={'PASS'},f'Explicit PASS verdict required: {path}: {lines}'
    return lines
