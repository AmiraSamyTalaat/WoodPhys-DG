# Commands

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_repo.py
python gate_audit/scripts/run_metadata_gate_audit.py
python scripts/stage4r/run_all_stage4r.py
```

For the data-dependent commands, first copy the locally held canonical inputs into `data/`:

```bash
python scripts/stage4r/run_all_stage4r.py --force
python scripts/run_all_stage4q.py
```

`--force` retrains and rewrites archived summary outputs. Keep a clean, tagged release unchanged and run full recomputation in a new local clone or separate working directory. Some analyses can take substantial time and may depend on the recorded software environment.

Source reconstruction (`scripts/run_stage4p.py`) writes `generated/` and requires canonical USDA and spruce inputs. Historical Stage 2C/3B scripts additionally require their original stage input ZIPs; their archived inputs are not part of the public release.
