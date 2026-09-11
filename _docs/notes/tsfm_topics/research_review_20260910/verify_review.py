"""Verify review links, preserved inputs and plotted numbers without training."""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import unquote
import csv
import hashlib
import json
import re
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
files = list((ROOT/'_docs/notes').rglob('*.md')) + list((ROOT/'_docs/history').glob('*.md'))
files += [ROOT/'README.md',ROOT/'_docs/PROJECT_LOG.md']
broken=[]; links=0; images=0
for path in files:
    body=path.read_text(encoding='utf-8-sig')
    assert body.count('```') % 2 == 0, path
    for match in re.finditer(r'(!?)\[[^\]\n]*\]\(([^\n]*?)\)',body):
        target=unquote(match[2].strip('<>')).split('#')[0]
        if not target or re.match(r'^(https?://|mailto:|codex:)',target):continue
        links+=1;images+=bool(match[1])
        if not (path.parent/target).exists():broken.append([str(path.relative_to(ROOT)),target])
assert not broken, broken
manifest=json.loads((HERE/'evidence/figure_sources.json').read_text())
for path,digest in manifest['sources_sha256'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
assert hashlib.sha256((HERE/'build_review.py').read_bytes()).hexdigest()==manifest['script_sha256']
relocations=json.loads((HERE/'evidence/relocation_manifest.json').read_text(encoding='utf-8'))['relocations']
assert len(relocations)==50
for entry in relocations:
    assert (ROOT/entry['new']).is_file() and not (ROOT/entry['old']).exists()

with (ROOT/'results/peft_fullft_reference_v3/metrics.csv').open(newline='',encoding='utf-8-sig') as f:
    raw=list(csv.DictReader(f))
with (HERE/'evidence/fullft_plot_values.csv').open(newline='',encoding='utf-8') as f:
    plotted=list(csv.DictReader(f))
errors=[]
assert len(raw)==44 and len(plotted)==6
for row in plotted:
    records=[r for r in raw if r['dataset']==row['dataset'] and r['procedure']=='SORT' and r['split']=='E']
    base=[float(r['score']) for r in records if r['arm']=='F0']
    scores=[float(r['score']) for r in records if r['arm']==row['arm']]
    assert len(base)==1 and len(scores)==3
    mean=sum(scores)/len(scores)
    errors.append(abs(mean-float(row['loss'])))
    errors.append(abs(100*(base[0]-mean)/base[0]-float(row['gain_pct_f0'])))
assert max(errors)<1e-10
decoded={}
for path in sorted((HERE/'figures').glob('*.png')):
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:decoded[path.name]=list(im.size)
assert len(decoded)==5
snapshot=json.loads((HERE/'evidence/hospital_partial_snapshot.json').read_text(encoding='utf-8'))
assert snapshot['completed_main_fits']==1 and snapshot['completed_eval_forecasts']==0
# The snapshot is historical: a later resumed run may now be complete.
report={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'markdown_files':len(files),
    'local_links_checked':links,'image_links_checked':images,'broken_local_links':broken,
    'moved_files':len(relocations),'figure_input_hashes_unchanged':len(manifest['sources_sha256']),
    'fullft_csv_records_read':len(raw),'plotted_aggregates_recomputed':len(plotted),
    'maximum_number_difference':max(errors),'decoded_figures':decoded,
    'historical_snapshot_hospital_status':'PAUSED_BY_USER_INCOMPLETE','new_model_training_or_inference':0,
    'scope':'Local link existence and saved-record consistency; not a new audit of original forecasts or public GitHub availability.'}
(HERE/'evidence/verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
