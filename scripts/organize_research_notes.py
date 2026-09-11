"""One-time, bounded relocation of numbered research notes and Markdown links."""
from pathlib import Path
import hashlib
import json
import os
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
TOP = ROOT / '_docs/notes/tsfm_topics'
AUDIT = TOP / 'research_review_20260910/evidence/relocation_manifest.json'
GROUPS = {
    '01_initial_topics': '초기 연구 후보: 토큰화·해상도·미래 공변량',
    '02_adaptation_scope': 'A: 적응 위치·메커니즘·외부 반복',
    '03_selection_calibration': 'B/C: 선택 안정성과 예측 보정',
    '04_objective_observation': '목적함수·관측 연산·집계 감독',
    '05_revision': '수정 정답과 자료 진입',
    '06_fullft_reference': '실제 Full FT 기준선과 메모리 복구',
    '07_research_direction': '연구 후보 스크린과 다음 방향',
    '08_hospital_shared_strength': 'Hospital 공유 LoRA 적용 강도',
}


def group(name):
    n = int(name[:2])
    if n <= 3:return '01_initial_topics'
    if n in [18,21,22,23] or 'topic_search_protocol' in name or 'next_method_literature' in name:return '07_research_direction'
    if n in [4,7,8,9,10,12,13]:return '02_adaptation_scope'
    if n in [5,6,11,14]:return '03_selection_calibration'
    if n in [15,16,17]:return '04_objective_observation'
    if n == 19:return '05_revision'
    if n == 20:return '06_fullft_reference'
    if n == 24:return '08_hospital_shared_strength'
    raise ValueError(name)


if AUDIT.exists():
    raise SystemExit('Already organized; inspect relocation_manifest.json instead of rerunning.')
mapping = {p.resolve():(p.parent/group(p.name)/p.name).resolve() for p in TOP.glob('*.md') if re.match(r'^\d{2}',p.name)}
assert mapping
for old,new in mapping.items():
    assert old.is_relative_to(TOP.resolve()) and new.is_relative_to(TOP.resolve())
    assert old.is_file() and not new.exists()

paths = list((ROOT/'_docs/notes').rglob('*.md')) + list((ROOT/'_docs/history').glob('*.md'))
paths += [ROOT/'README.md',ROOT/'_docs/PROJECT_LOG.md']
original = {p.resolve():p.read_text(encoding='utf-8-sig') for p in paths}
hashes = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in original}
by_name = {p.name:p for p in mapping}
link_re = re.compile(r'(!?\[[^\]\n]*\]\()([^\n]*?)(\))')
changed = []


def target_path(raw, parent):
    q = unquote(raw.replace('\\','/'))
    if q.startswith('file:///'):q=q[8:]
    if re.match(r'^[A-Za-z]:/',q):return Path(q).resolve()
    return (parent/q).resolve()


for old,content in original.items():
    new=mapping.get(old,old)
    def replace(match):
        raw=match[2]
        if re.match(r'^(https?://|mailto:|#|codex:)',raw):return match[0]
        raw=raw.strip('<>')
        pathpart,sep,fragment=raw.partition('#')
        target=target_path(pathpart,old.parent)
        if not target.exists():
            # Repair pre-existing too-shallow results links and bare note links.
            normalized=pathpart.replace('\\','/')
            if '/results/' in normalized:
                candidate=ROOT/'results'/normalized.split('/results/',1)[1]
                if candidate.exists():target=candidate.resolve()
            elif Path(pathpart).name in by_name:target=by_name[Path(pathpart).name]
        if not target.exists() and target not in mapping:return match[0]
        target=mapping.get(target,target)
        relative=os.path.relpath(target,new.parent).replace('\\','/')
        return match[1]+relative+(sep+fragment if sep else '')+match[3]
    content=link_re.sub(replace,content)
    # Update explicit repository paths in prose; source/run hash contracts stay untouched.
    for src,dst in mapping.items():
        content=content.replace(src.relative_to(ROOT).as_posix(),dst.relative_to(ROOT).as_posix())
        content=content.replace(str(src.relative_to(ROOT)),str(dst.relative_to(ROOT)))
    if new!=old or content!=original[old]:
        new.parent.mkdir(parents=True,exist_ok=True)
        if new!=old:old.rename(new)
        new.write_text(content,encoding='utf-8')
        changed.append(new.relative_to(ROOT).as_posix())

for folder,title in GROUPS.items():
    directory=TOP/folder
    files=sorted(directory.glob('*.md'))
    body=f'# {title}\n\n[전체 정리](../research_review_20260910/README.md) · [전체 목록](../README.md)\n\n'
    for p in files:
        heading=p.read_text(encoding='utf-8-sig').splitlines()[0].lstrip('# ').strip()
        body+=f'- [{heading}]({p.name})\n'
    (directory/'README.md').write_text(body,encoding='utf-8')

audit={'scope':'Markdown notes, history, PROJECT_LOG and root README only; no experiment/run/result mutation',
       'relocations':[{'old':p.relative_to(ROOT).as_posix(),'new':q.relative_to(ROOT).as_posix(),
                       'before_sha256':hashes[p],'after_sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for p,q in mapping.items()],
       'changed_markdown':changed}
AUDIT.write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'moved':len(mapping),'updated_markdown':len(changed),'groups':len(GROUPS)}))
