from pathlib import Path
import csv,hashlib,json

ROOT=Path(__file__).resolve().parents[1]
REQUIRED={
  'PINUS_MODEL_BENCHMARK_SUMMARY.csv':12,
  'PINUS_LABEL_EFFICIENCY_SUMMARY.csv':24,
  'PINUS_RISK_SUMMARY.csv':2,
  'PINUS_CONFORMAL_SUMMARY.csv':12,
  'PINUS_CONSENSUS_PERFORMANCE.csv':6,
}

def main():
  out=[]
  for fname,n in REQUIRED.items():
    p=ROOT/'tables'/'stage4r'/fname
    if not p.is_file():raise SystemExit('Missing '+str(p))
    with p.open(newline='',encoding='utf-8-sig') as file:
      lines=sum(1 for _ in csv.DictReader(file))
    if lines<n:raise SystemExit(f'{fname}: expected >= {n}, found {lines}')
    out.append({'file':fname,'rows':lines,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  expected=ROOT/'gate_audit'/'inputs'/'regime_metadata.csv'
  with expected.open(newline='',encoding='utf-8-sig') as file:
    fields=csv.DictReader(file).fieldnames
  if any(x in fields for x in ('MOE','MOR','moe_mpa','mor_mpa','target','y_true')):
    raise SystemExit('Gate metadata must exclude target outcomes')
  print('ARCHIVED SUMMARY FILES: PASS')
  for x in out: print(x['file'],x['rows'])
  print('GATE METADATA SCHEMA: PASS')

if __name__=='__main__':main()
