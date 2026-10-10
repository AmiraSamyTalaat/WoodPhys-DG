from __future__ import annotations
from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
scripts=[
 'run_stage4q_transport_or_adapt.py',
 'run_stage4q_extra_models.py',
 'run_stage4q_target_conformal.py',
 'run_stage4q_guard_stability.py',
 'run_stage4q_consensus_guard.py',
]
for name in scripts:
    print(f'=== {name} ===',flush=True)
    subprocess.run([sys.executable,str(ROOT/'scripts'/name)],check=True,cwd=ROOT)
print('STAGE4Q_ALL_ANALYSES_PASS')
