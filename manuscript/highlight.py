"""Build a version of the revised manuscript with changes highlighted relative to the submitted version."""
import json, re, difflib
old = json.load(open('/home/claude/work/manuscript/manuscript.json'))['blocks']
new = json.load(open('manuscript_rev.json'))['blocks']
key = lambda t: re.sub(r'\^\{[\d,–]+\}', '^{#}', t)
TEXT = ('p', 'pleft', 'h1', 'h2', 'title', 'center')
old_txt = [b.get('text') or b.get('caption') or '' for b in old]
old_refs = {re.sub(r'^\d+\. ', '', b['text']) for b in old if b['type'] == 'ref'}
def mark(t): return '==' + t + '==' if t.strip() else t
def diff(o, n):
    ow, nw = o.split(' '), n.split(' ')
    sm = difflib.SequenceMatcher(None, [key(w) for w in ow], [key(w) for w in nw], autojunk=False)
    out = []
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        seg = ' '.join(nw[b1:b2])
        out.append(seg if op == 'equal' else (mark(seg) if op in ('replace', 'insert') else ''))
    return ' '.join(x for x in out if x)
def best(t):
    k = key(t); r = [(difflib.SequenceMatcher(None, key(o), k, autojunk=False).quick_ratio(), i) for i, o in enumerate(old_txt)]
    r.sort(reverse=True)
    cands = [(difflib.SequenceMatcher(None, key(old_txt[i]), k, autojunk=False).ratio(), i) for _, i in r[:5]]
    return max(cands)
nchg = 0
for b in new:
    fld = 'text' if 'text' in b else ('caption' if 'caption' in b else None)
    if not fld: continue
    t = b[fld]
    if b['type'] == 'ref':
        if re.sub(r'^\d+\. ', '', t) not in old_refs: b[fld] = mark(t); nchg += 1
        continue
    sc, i = best(t)
    if sc > 0.999: continue
    b[fld] = diff(old_txt[i], t) if sc >= 0.45 else mark(t); nchg += 1
    if b['type'] == 'table':  # new tables: highlight cells too
        if sc < 0.45: b['rows'] = [[mark(str(c)) for c in r] for r in b['rows']]
json.dump({'blocks': new}, open('manuscript_rev_highlighted.json', 'w'))
print('changed blocks', nchg)
