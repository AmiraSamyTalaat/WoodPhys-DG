from __future__ import annotations
from pathlib import Path
import hashlib, json, sys, urllib.request

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'public_clearwood_raw'
RAW.mkdir(parents=True, exist_ok=True)

SOURCES = {
    'LPF_SFB_BRAZILIAN_WOODS': {
        'url': 'https://dados.florestal.gov.br/dataset/a1c2f38f-f2b9-4b17-81f8-6b57bdefe78c/resource/cb533446-ae2f-4767-a2e5-676273c85852/download/dados_abertos_banco_de_dados_de_madeiras_brasileiras_lpfsfb.csv',
        'filename': 'lpf_sfb_brazilian_woods.csv',
        'license': 'Creative Commons Attribution',
        'role': 'PRIMARY_SAME_REGIME_CANDIDATE_PENDING_SCHEMA_AND_OVERLAP_AUDIT',
    },
    'SCIELO_BRAZILIAN_HARDWOODS': {
        'url': 'https://data.scielo.org/api/access/datafile/:persistentId?persistentId=doi:10.48331/SCIELODATA.AZSXAM/ROBOW4',
        'filename': 'scielo_brazilian_hardwoods.tab',
        'license': 'CC BY 4.0',
        'role': 'SECONDARY_STANDARD_STRATIFIED_SENSITIVITY_ONLY',
    },
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

results = []
for name, meta in SOURCES.items():
    out = RAW / meta['filename']
    status = 'DOWNLOADED'
    error = ''
    try:
        req = urllib.request.Request(meta['url'], headers={'User-Agent': 'Mozilla/5.0 WoodPhys-DG academic reproducibility'})
        with urllib.request.urlopen(req, timeout=90) as response, out.open('wb') as f:
            f.write(response.read())
        if out.stat().st_size == 0:
            raise RuntimeError('Downloaded file is empty')
    except Exception as exc:
        status = 'DOWNLOAD_FAILED'
        error = repr(exc)
    results.append({
        'dataset': name,
        **meta,
        'status': status,
        'path': str(out.relative_to(ROOT)) if out.exists() else '',
        'size_bytes': out.stat().st_size if out.exists() else 0,
        'sha256': sha256(out) if out.exists() else '',
        'error': error,
    })

report = ROOT / 'dataset_registry' / 'PUBLIC_CLEARWOOD_DOWNLOAD_REPORT.json'
report.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(results, indent=2, ensure_ascii=False))
if any(r['status'] != 'DOWNLOADED' for r in results):
    sys.exit(2)
