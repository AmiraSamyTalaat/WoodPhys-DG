
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
import urllib.request
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input" / "STAGE3A_VERIFIED_YELLOW_RESULT.zip"
OUT = ROOT / "stage3a_fix2_output"
WORK = OUT / "_work"
FREEZE = OUT / "method_freeze_preserved"
QUARANTINE = OUT / "quarantine"
REPORTS = OUT / "reports"

ARTICLE_ID = 12241268
PAGE_URL = (
    "https://scholardata.sun.ac.za/articles/dataset/"
    "Flexural_properties_of_Pinus_patula_planted_at_different_initial_spacing/"
    "12241268"
)
API_URLS = [
    f"https://api.figshare.com/v2/articles/{ARTICLE_ID}",
    f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/files",
]

LOG_LINES = []
ERROR_LINES = []

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def log(message):
    line = f"[{now()}] {message}"
    LOG_LINES.append(line)
    print(line, flush=True)

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def fetch(url, max_mb=100, attempts=4):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/148 Safari/537.36"
        ),
        "Accept": "*/*"
    }
    last = None
    for attempt in range(1, attempts+1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response:
                data = response.read(max_mb*1024*1024+1)
                content_type = response.headers.get("Content-Type", "")
            if len(data) > max_mb*1024*1024:
                raise RuntimeError(f"Object exceeds {max_mb} MB")
            if len(data) < 20:
                raise RuntimeError(f"Object too small: {len(data)}")
            low = data[:500].lower()
            if b"<html" in low or b"access denied" in low or b"nosuchkey" in low:
                raise RuntimeError("HTML/error page returned instead of data")
            return data, content_type
        except Exception as exc:
            last = exc
            log(f"Attempt {attempt}/{attempts} failed: {url} :: {exc}")
    raise RuntimeError(str(last))

def verify_stage3a():
    dst = WORK / "stage3a"
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(INPUT) as z:
        z.extractall(dst)

    status = json.loads((dst/"STATUS.json").read_text(encoding="utf-8"))
    success = json.loads((dst/"SUCCESS_MARKER.json").read_text(encoding="utf-8"))
    verify = json.loads((dst/"OUTPUT_VERIFICATION.json").read_text(encoding="utf-8"))
    decision = json.loads(
        (dst/"SCIENTIFIC_STAGE3A_DECISION.json").read_text(encoding="utf-8")
    )

    if status.get("status") != "completed" or not success.get("success"):
        raise RuntimeError("Stage3A input is incomplete")
    if not verify.get("verification_passed"):
        raise RuntimeError("Stage3A verification failed")
    if not verify.get("method_freeze_created_before_download"):
        raise RuntimeError("Method-freeze chronology not verified")
    if decision.get("decision") != "YELLOW_LOCKBOX_ACCESS_BLOCKED":
        raise RuntimeError(
            "Expected YELLOW_LOCKBOX_ACCESS_BLOCKED; got "
            + str(decision.get("decision"))
        )

    hash_rows = pd.read_csv(dst/"HASHES_SHA256.csv")
    mismatches = []
    for _, row in hash_rows.iterrows():
        path = dst / row["relative_path"]
        actual = sha256(path) if path.exists() else "MISSING"
        if actual != row["sha256"]:
            mismatches.append(row["relative_path"])
    if mismatches:
        raise RuntimeError(f"Stage3A hash mismatches: {mismatches[:5]}")

    shutil.copytree(dst/"method_freeze", FREEZE, dirs_exist_ok=True)
    log("Verified Stage3A and preserved the original method freeze")
    return dst

def create_replacement_amendment():
    amendment = {
        "amendment_id": "WOODPHYS_DG_LOCKBOX_REPLACEMENT_001",
        "original_candidate": {
            "source": "Eucalyptus grandis SciELO Figshare 14321961",
            "status": "REJECTED_BEFORE_EXTERNAL_PREDICTION",
            "reason": (
                "Repository file objects return NoSuchKey/AccessDenied; published "
                "tables are aggregated and predictor compatibility is not established."
            ),
            "external_predictions_computed": False,
            "external_metrics_computed": False
        },
        "replacement_candidate": {
            "source": "Pinus patula SUNScholarData 12241268",
            "doi": "10.25413/sun.12241268",
            "page_url": PAGE_URL
        },
        "method_parameters_unchanged": True,
        "frozen_parameters": {
            "moe_mpa": {
                "applicability_temperature": 3.0,
                "uq_beta": 0.75
            },
            "mor_mpa": {
                "applicability_temperature": 0.2,
                "uq_beta": 1.0
            }
        },
        "created_before_replacement_download": True,
        "created_at": now()
    }
    write_json(FREEZE/"LOCKBOX_REPLACEMENT_AMENDMENT.json", amendment)
    log("Created replacement amendment before accessing Pinus patula files")

def inspect_file(path):
    suffix = path.suffix.lower()
    frames = []
    if suffix == ".xlsx":
        book = pd.ExcelFile(path, engine="openpyxl")
        frames = [
            (sheet, pd.read_excel(path, sheet_name=sheet, engine="openpyxl", header=None))
            for sheet in book.sheet_names
        ]
    elif suffix == ".xls":
        book = pd.ExcelFile(path, engine="xlrd")
        frames = [
            (sheet, pd.read_excel(path, sheet_name=sheet, engine="xlrd", header=None))
            for sheet in book.sheet_names
        ]
    elif suffix in {".csv", ".tsv", ".txt"}:
        frame = pd.read_csv(path, sep=None, engine="python", header=None)
        frames = [("tabular", frame)]
    else:
        return [], None

    audits = []
    best = None
    for sheet, frame in frames:
        best_header = 0
        best_score = -1
        for idx in range(min(30, len(frame))):
            vals = [
                re.sub(r"[^a-z0-9]+", " ", str(x or "").lower()).strip()
                for x in frame.iloc[idx].tolist()
            ]
            joined = " | ".join(vals)
            score = sum(term in joined for term in [
                "moe", "modulus of elasticity",
                "mor", "modulus of rupture",
                "density", "specific gravity",
                "wood density", "knot", "microfibril",
                "spacing", "board", "lumber"
            ])
            if score > best_score:
                best_score = score
                best_header = idx

        headers = [
            str(x).strip() if str(x).strip() not in {"", "nan"} else f"COL_{i+1}"
            for i, x in enumerate(frame.iloc[best_header].tolist())
        ]
        joined = " | ".join(
            re.sub(r"[^a-z0-9]+", " ", x.lower()).strip() for x in headers
        )
        rows = frame.iloc[best_header+1:].dropna(how="all")

        audit = {
            "file": path.name,
            "sheet": sheet,
            "header_row_1_based": best_header+1,
            "rows_after_header": int(len(rows)),
            "column_count": int(len(headers)),
            "headers": json.dumps(headers, ensure_ascii=False),
            "has_moe": (
                "moe" in joined or "modulus of elasticity" in joined
            ),
            "has_mor": (
                "mor" in joined or "modulus of rupture" in joined
            ),
            "has_density_or_specific_gravity": any(term in joined for term in [
                "density", "specific gravity", "wood density",
                "basic density", "air dry density"
            ]),
            "has_spacing_or_provenance": any(term in joined for term in [
                "spacing", "planting", "site", "provenance", "treatment"
            ]),
            "sha256": sha256(path)
        }
        audits.append(audit)
        rank = (
            int(audit["has_moe"]) + int(audit["has_mor"]) +
            int(audit["has_density_or_specific_gravity"]),
            audit["rows_after_header"]
        )
        if best is None or rank > best[0]:
            best = (rank, audit)
    return audits, best[1] if best else None

def acquire():
    access = []
    metadata = None
    files = []

    for url in API_URLS:
        try:
            data, content_type = fetch(url, max_mb=10)
            name = (
                "ARTICLE_METADATA.json"
                if url.endswith(str(ARTICLE_ID))
                else "FILE_LIST.json"
            )
            (REPORTS/name).write_bytes(data)
            parsed = json.loads(data.decode("utf-8"))
            if isinstance(parsed, dict):
                metadata = parsed
                files += parsed.get("files", []) or []
            elif isinstance(parsed, list):
                files += parsed
            access.append({
                "route": url, "status": "SUCCESS",
                "bytes": len(data), "content_type": content_type, "error": ""
            })
        except Exception as exc:
            access.append({
                "route": url, "status": "FAILED",
                "bytes": 0, "content_type": "", "error": str(exc)
            })

    dedup = {}
    for item in files:
        key = str(item.get("id") or item.get("download_url") or item.get("name"))
        dedup[key] = item
    files = list(dedup.values())

    downloaded = []
    for item in files:
        name = str(item.get("name") or "")
        suffix = Path(name).suffix.lower()
        if suffix not in {".csv", ".tsv", ".txt", ".xls", ".xlsx"}:
            continue
        size = int(item.get("size") or 0)
        if size and size > 100*1024*1024:
            continue

        urls = []
        if item.get("download_url"):
            urls.append(item["download_url"])
        if item.get("id"):
            urls += [
                f"https://figshare.com/ndownloader/files/{item['id']}",
                f"https://scholardata.sun.ac.za/ndownloader/files/{item['id']}"
            ]

        for url in urls:
            try:
                data, content_type = fetch(url, max_mb=100)
                path = QUARANTINE / name
                path.write_bytes(data)
                downloaded.append(path)
                access.append({
                    "route": url, "status": "DOWNLOADED_TO_QUARANTINE",
                    "bytes": len(data), "content_type": content_type, "error": ""
                })
                break
            except Exception as exc:
                access.append({
                    "route": url, "status": "FAILED",
                    "bytes": 0, "content_type": "", "error": str(exc)
                })

    pd.DataFrame(access).to_csv(
        REPORTS/"SOURCE_ACCESS_AUDIT.csv",
        index=False, encoding="utf-8-sig"
    )
    return metadata, downloaded

def make_upload():
    upload = OUT/"UPLOAD_THIS_STAGE3A_FIX2_RESULT.zip"
    with zipfile.ZipFile(upload, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path != upload and "_work" not in path.parts:
                z.write(path, path.relative_to(OUT))

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in [OUT, WORK, FREEZE, QUARANTINE, REPORTS]:
        d.mkdir(parents=True, exist_ok=True)

    write_json(OUT/"STATUS.json", {
        "stage": "STAGE3A_FIX2", "version": "1.0.0",
        "status": "running", "started_at": now()
    })

    try:
        verify_stage3a()
        create_replacement_amendment()
        metadata, downloaded = acquire()

        audits = []
        best_candidates = []
        for path in downloaded:
            try:
                file_audits, best = inspect_file(path)
                audits.extend(file_audits)
                if best:
                    best_candidates.append(best)
            except Exception as exc:
                audits.append({
                    "file": path.name,
                    "sheet": "",
                    "rows_after_header": 0,
                    "has_moe": False,
                    "has_mor": False,
                    "has_density_or_specific_gravity": False,
                    "inspection_error": str(exc),
                    "sha256": sha256(path)
                })

        pd.DataFrame(audits).to_csv(
            REPORTS/"LOCKBOX_SCHEMA_AUDIT.csv",
            index=False, encoding="utf-8-sig"
        )

        max_rows = max(
            [int(x.get("rows_after_header", 0)) for x in audits] or [0]
        )
        has_moe = any(bool(x.get("has_moe")) for x in audits)
        has_mor = any(bool(x.get("has_mor")) for x in audits)
        has_density = any(
            bool(x.get("has_density_or_specific_gravity")) for x in audits
        )

        license_name = ""
        if isinstance(metadata, dict):
            lic = metadata.get("license")
            if isinstance(lic, dict):
                license_name = str(lic.get("name") or lic.get("url") or "")
            elif lic:
                license_name = str(lic)

        if not downloaded:
            decision = "YELLOW_PINUS_LOCKBOX_ACCESS_BLOCKED"
            rationale = (
                "The replacement candidate is registered, but no tabular file "
                "could be downloaded automatically."
            )
        elif has_moe and has_mor and has_density and max_rows >= 50:
            decision = "GREEN_PINUS_LOCKBOX_REGISTERED_AND_QUARANTINED"
            rationale = (
                "An independent Pinus patula tabular file with MOE, MOR, a "
                "density-like predictor, and at least 50 rows was quarantined. "
                "No predictions or external metrics were computed."
            )
        elif has_moe and has_mor and max_rows >= 50 and not has_density:
            decision = "YELLOW_PINUS_LOCKBOX_PREDICTOR_GAP"
            rationale = (
                "The files include MOE and MOR but no compatible density or "
                "specific-gravity predictor was detected."
            )
        else:
            decision = "YELLOW_PINUS_LOCKBOX_SCHEMA_INSUFFICIENT"
            rationale = (
                "Files were downloaded, but the schema is too small, aggregated, "
                "or lacks required outcomes."
            )

        scientific = {
            "decision": decision,
            "rationale": rationale,
            "method_freeze_integrity": "PASS",
            "replacement_amendment_created_before_download": True,
            "external_predictions_computed": False,
            "external_metrics_computed": False,
            "quarantine_files": [
                {
                    "file": p.name,
                    "size_bytes": p.stat().st_size,
                    "sha256": sha256(p)
                }
                for p in downloaded
            ],
            "license_metadata": license_name,
            "best_schema_candidates": best_candidates,
            "next_step": (
                "STAGE3B_FROZEN_EXTERNAL_EVALUATION"
                if decision == "GREEN_PINUS_LOCKBOX_REGISTERED_AND_QUARANTINED"
                else "MANUAL_BROWSER_RECOVERY_OR_NEW_CANDIDATE"
            ),
            "generated_at": now()
        }
        write_json(OUT/"SCIENTIFIC_STAGE3A_FIX2_DECISION.json", scientific)

        report = [
            "# WOODPHYS_DG_Q1 — Stage 3A FIX2",
            "",
            f"## القرار: **{decision}**",
            "",
            rationale,
            "",
            "## الحماية العلمية",
            "- Method freeze preserved: PASS",
            "- Replacement amendment created before download: YES",
            "- Frozen parameters changed: NO",
            "- External predictions computed: NO",
            "- External metrics computed: NO",
            f"- Quarantine files: {len(downloaded)}",
            "",
        ]
        for row in best_candidates:
            report += [
                f"### {row['file']} / {row['sheet']}",
                f"- Rows after header: {row['rows_after_header']}",
                f"- MOE detected: {row['has_moe']}",
                f"- MOR detected: {row['has_mor']}",
                "- Density/specific gravity detected: "
                f"{row['has_density_or_specific_gravity']}",
                "",
            ]
        report += [
            "## التالي",
            (
                "إذا كان القرار أخضر، ارفعي الحزمة فقط ولا تفتحي quarantine. "
                "سنبني Stage 3B من الطريقة المجمدة دون تعديل."
            )
        ]
        (OUT/"PINUS_LOCKBOX_REPORT_AR.md").write_text(
            "\n".join(report)+"\n", encoding="utf-8"
        )

        write_json(OUT/"OUTPUT_VERIFICATION.json", {
            "verification_passed": True,
            "stage3a_verified": True,
            "method_freeze_preserved": True,
            "replacement_registered_before_download": True,
            "external_predictions_computed": False,
            "external_metrics_computed": False,
            "quarantine_files": len(downloaded)
        })
        write_json(OUT/"STATUS.json", {
            "stage": "STAGE3A_FIX2", "version": "1.0.0",
            "status": "completed", "decision": decision,
            "completed_at": now()
        })
        write_json(OUT/"SUCCESS_MARKER.json", {
            "success": True, "stage": "STAGE3A_FIX2", "version": "1.0.0",
            "decision": decision, "timestamp": now()
        })
        (OUT/"ERROR_LOG.txt").write_text("", encoding="utf-8")
        (OUT/"RUN_LOG.txt").write_text(
            "\n".join(LOG_LINES)+"\n", encoding="utf-8"
        )

        manifest, hashes = [], []
        for path in sorted(OUT.rglob("*")):
            if not path.is_file() or "_work" in path.parts:
                continue
            rel = path.relative_to(OUT).as_posix()
            if rel in {
                "FILE_MANIFEST.csv", "HASHES_SHA256.csv",
                "UPLOAD_THIS_STAGE3A_FIX2_RESULT.zip"
            }:
                continue
            manifest.append({
                "relative_path": rel,
                "size_bytes": path.stat().st_size
            })
            hashes.append({
                "relative_path": rel,
                "sha256": sha256(path)
            })
        pd.DataFrame(manifest).to_csv(
            OUT/"FILE_MANIFEST.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(hashes).to_csv(
            OUT/"HASHES_SHA256.csv", index=False, encoding="utf-8-sig"
        )
        make_upload()

        log(f"Completed Stage3A FIX2. Decision: {decision}")
        (OUT/"RUN_LOG.txt").write_text(
            "\n".join(LOG_LINES)+"\n", encoding="utf-8"
        )
        return 0

    except Exception as exc:
        ERROR_LINES.append(traceback.format_exc())
        write_json(OUT/"STATUS.json", {
            "stage": "STAGE3A_FIX2", "status": "failed",
            "error": str(exc), "completed_at": now()
        })
        write_json(OUT/"SUCCESS_MARKER.json", {
            "success": False, "stage": "STAGE3A_FIX2",
            "error": str(exc), "timestamp": now()
        })
        (OUT/"RUN_LOG.txt").write_text(
            "\n".join(LOG_LINES)+"\n", encoding="utf-8"
        )
        (OUT/"ERROR_LOG.txt").write_text(
            "\n".join(ERROR_LINES)+"\n", encoding="utf-8"
        )
        try:
            make_upload()
        except Exception:
            pass
        print(ERROR_LINES[-1], file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
