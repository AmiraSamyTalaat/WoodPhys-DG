from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
OUT = BASE / 'tables' / 'stage4r'
REQUIRED = {
    'PINUS_MODEL_BENCHMARK_SUMMARY.csv': 12,
    'PINUS_LABEL_EFFICIENCY_SUMMARY.csv': 24,
    'PINUS_RISK_SUMMARY.csv': 2,
    'PINUS_CONFORMAL_SUMMARY.csv': 12,
    'PINUS_CONSENSUS_PERFORMANCE.csv': 6,
}

def verify() -> None:
    failures=[]
    for name,min_rows in REQUIRED.items():
        path=OUT/name
        if not path.exists():
            failures.append(f'missing {name}');continue
        rows=len(pd.read_csv(path))
        if rows < min_rows: failures.append(f'{name}: expected at least {min_rows} rows, found {rows}')
    if failures:
        raise SystemExit('Stage 4R verification failed: ' + '; '.join(failures))
    print('Stage 4R outputs verified successfully.')
    print(pd.read_csv(OUT/'PINUS_MODEL_BENCHMARK_SUMMARY.csv').sort_values(['target','median_r2'],ascending=[True,False]).groupby('target').head(2).to_string(index=False))
    print(pd.read_csv(OUT/'PINUS_RISK_SUMMARY.csv').to_string(index=False))
    print(pd.read_csv(OUT/'PINUS_CONSENSUS_PERFORMANCE.csv').to_string(index=False))

def recompute() -> None:
    for script in ['run_stage4r_pinus_benchmark.py','run_stage4r_pinus_risk_fast.py']:
        print(f'Running {script}...', flush=True)
        subprocess.run([sys.executable, str(ROOT/script)], check=True)
    verify()

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--force',action='store_true',help='Recompute all Stage 4R outputs before verification.')
    args=ap.parse_args()
    recompute() if args.force else verify()
