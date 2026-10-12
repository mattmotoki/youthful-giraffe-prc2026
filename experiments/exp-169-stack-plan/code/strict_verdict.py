"""Unit 024 review gate: explicit, unquoted verdicts, including conflicting forms."""
import re
from pathlib import Path

def parse_verdicts(text):
    out=[];fenced=False
    for number,line in enumerate(text.splitlines(),1):
        line=line.strip()
        if line.startswith(('```','~~~')):fenced=not fenced;continue
        if fenced or line.startswith('>'):continue
        plain=line.replace('**','').replace('__','').lstrip('# ').strip()
        plain=re.sub(r'^(?:[-*+]\s+|\d+[.)]\s+)', '', plain)
        hit=re.match(r'^(?:(?:final\s+)?verdict|fresh-context correctness review)\s*:\s*(PASS|FAIL|BLOCK|REJECT)(?=$|[\s.,;])',plain,re.I)
        if hit:out.append({'line':number,'verdict':hit[1].upper(),'text':line})
    return out

def require_strict_pass(path):
    result=parse_verdicts(Path(path).read_text())
    assert result and {r['verdict'] for r in result}=={'PASS'},f'Explicit unopposed PASS required: {path}: {result}'
    return result
