from __future__ import annotations
from pathlib import Path
import argparse, hashlib, json, re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def norm(s: str) -> str:
    s = str(s).strip().lower()
    return re.sub(r'[^a-z0-9]+', '_', s).strip('_')

def sha256(path: Path) -> str:
    h=hashlib.sha256();
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()

def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {'.tab','.tsv'}:
        return pd.read_csv(path, sep='\t')
    if path.suffix.lower() in {'.xlsx','.xls'}:
        return pd.read_excel(path)
    for enc in ['utf-8-sig','utf-8','latin-1']:
        for sep in [',',';','\t']:
            try:
                d=pd.read_csv(path,encoding=enc,sep=sep)
                if d.shape[1] > 1:return d
            except Exception:pass
    raise RuntimeError(f'Could not parse {path}')

p=argparse.ArgumentParser()
p.add_argument('input_file')
p.add_argument('--dataset-id',required=True)
a=p.parse_args()
path=Path(a.input_file)
df=read_any(path)
original=list(df.columns)
df.columns=[norm(c) for c in df.columns]

patterns={
 'record_id':['record_id','id','codigo','code','numero'],
 'species':['scientific_name','species','especie','nome_cientifico','taxon'],
 'genus':['genus','genero'],
 'specific_gravity':['specific_gravity','basic_specific_gravity','densidade_basica','basic_density','dbas'],
 'density_kg_m3':['density','density_kg_m3','densidade','massa_especifica'],
 'moe_mpa':['moe_mpa','moe','modulo_de_elasticidade','modulus_of_elasticity','elasticidade'],
 'mor_mpa':['mor_mpa','mor','modulo_de_ruptura','modulus_of_rupture','ruptura'],
 'moisture_percent':['moisture_percent','moisture','umidade'],
 'test_standard':['test_standard','standard','norma','metodo'],
 'source_reference':['source_reference','reference','referencia','fonte','paper'],
}
found={}
for role,cands in patterns.items():
    hits=[]
    for c in df.columns:
        if c in cands or any(x in c for x in cands):hits.append(c)
    found[role]=hits
report={
 'dataset_id':a.dataset_id,'input_file':str(path),'sha256':sha256(path),'rows':len(df),
 'original_columns':original,'normalized_columns':list(df.columns),'candidate_mapping':found,
 'automatic_model_run_authorized':False,
 'required_human_checks':['testing standard and specimen regime','units','USDA/source overlap','row independence','outcome columns withheld until compatibility decision is frozen'],
}
out=ROOT/'dataset_registry'/f'{a.dataset_id}_SCHEMA_REPORT.json'
out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=False))
