#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/stage4r/run_all_stage4r.py --force
