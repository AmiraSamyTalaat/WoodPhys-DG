
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
INPUT_ZIP = ROOT / "input" / "STAGE0C_VERIFIED_INPUT.zip"
OUT = ROOT / "stage0d_output"
WORK = OUT / "_work"
RAW = OUT / "raw"
DATA = OUT / "data"
REPORTS = OUT / "reports"
TAX = OUT / "taxonomy"
LOG_LINES: list[str] = []
ERROR_LINES: list[str] = []

URLS = [
    "https://editors.eol.org/uploaded_resources/d1e/2b0/archive.zip",
    "https://zenodo.org/records/13322441/files/archive.zip?download=1",
]

def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def log(message: str) -> None:
    line = f"[{now()}] {message}"
    LOG_LINES.append(line)
    print(line, flush=True)

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip()).lower()

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def validate_stage0c() -> Path:
    if not INPUT_ZIP.exists():
        raise FileNotFoundError(f"Missing canonical input: {INPUT_ZIP}")
    target = WORK / "stage0c"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(INPUT_ZIP) as z:
        z.extractall(target)
    required = ["SUCCESS_MARKER.json", "OUTPUT_VERIFICATION.json",
                "SCIENTIFIC_GATE_DECISION.json",
                "taxonomy/GBIF_TAXONOMY_CANDIDATES.csv"]
    missing = [x for x in required if not (target / x).exists()]
    if missing:
        raise RuntimeError(f"Stage0C input missing: {missing}")
    success = json.loads((target / "SUCCESS_MARKER.json").read_text(encoding="utf-8"))
    verify = json.loads((target / "OUTPUT_VERIFICATION.json").read_text(encoding="utf-8"))
    if not success.get("success") or not verify.get("verification_passed"):
        raise RuntimeError("Stage0C input is not technically verified")
    log("Verified Stage0C canonical input")
    return target

def taxonomy_correction(stage0c: Path) -> dict:
    src = stage0c / "taxonomy" / "GBIF_TAXONOMY_CANDIDATES.csv"
    rows = list(csv.DictReader(src.open("r", encoding="utf-8-sig", newline="")))
    out_rows = []
    counts = {
        "total": len(rows), "verified_species": 0, "verified_genus": 0,
        "manual_review": 0, "unresolved_none": 0, "canonical_nonempty": 0
    }
    for r in rows:
        canonical = (r.get("gbif_canonical_name") or "").strip()
        match_type = (r.get("gbif_match_type") or "").upper().strip()
        rank = (r.get("gbif_rank") or "").upper().strip()
        status = (r.get("gbif_status") or "").upper().strip()
        try:
            confidence = int(float(r.get("gbif_confidence") or 0))
        except Exception:
            confidence = 0

        if canonical:
            counts["canonical_nonempty"] += 1

        valid_status = status in {"ACCEPTED", "SYNONYM"}
        valid_match = match_type in {"EXACT", "FUZZY"}
        if canonical and valid_status and valid_match and rank == "SPECIES" and confidence >= 90:
            corrected = "VERIFIED_SPECIES_CANDIDATE"
            counts["verified_species"] += 1
        elif canonical and valid_status and valid_match and rank == "GENUS" and confidence >= 90:
            corrected = "VERIFIED_GENUS_CANDIDATE"
            counts["verified_genus"] += 1
        elif match_type == "NONE" or not canonical:
            corrected = "UNRESOLVED_COMMON_NAME_OR_NONE"
            counts["unresolved_none"] += 1
        else:
            corrected = "MANUAL_REVIEW"
            counts["manual_review"] += 1
        rr = dict(r)
        rr["corrected_resolution_decision"] = corrected
        rr["stage0c_confidence_not_used_alone"] = "YES"
        out_rows.append(rr)

    TAX.mkdir(parents=True, exist_ok=True)
    fields = list(out_rows[0].keys()) if out_rows else []
    with (TAX / "GBIF_TAXONOMY_CORRECTED_AUDIT.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    unresolved = [r for r in out_rows if r["corrected_resolution_decision"] != "VERIFIED_SPECIES_CANDIDATE"]
    with (TAX / "TAXONOMY_MANUAL_QUEUE.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(unresolved)
    write_json(TAX / "TAXONOMY_CORRECTION_SUMMARY.json", counts)
    log(
        "Corrected taxonomy metric: "
        f"species={counts['verified_species']}, genus={counts['verified_genus']}, "
        f"unresolved NONE={counts['unresolved_none']}, manual={counts['manual_review']}"
    )
    return counts

def download(url: str, dest: Path, attempts: int = 3) -> None:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 Chrome/148 Safari/537.36",
        "Accept": "application/zip,application/octet-stream,*/*",
    }
    last = None
    for i in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) < 10000:
                raise RuntimeError(f"Downloaded object is too small: {len(data)} bytes")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            if not zipfile.is_zipfile(tmp):
                preview = data[:200].decode("utf-8", errors="replace")
                raise RuntimeError(f"Not a valid ZIP; prefix={preview!r}")
            tmp.replace(dest)
            return
        except Exception as e:
            last = e
            log(f"Download attempt {i}/{attempts} failed: {url} :: {e}")
            time.sleep(i * 2)
    raise RuntimeError(f"All attempts failed for {url}: {last}")

def acquire_wwes() -> tuple[Path | None, list[dict]]:
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / "WWES_EO_L_DARWIN_CORE_ARCHIVE.zip"
    audit = []
    if dest.exists() and zipfile.is_zipfile(dest):
        log(f"Using existing validated WWES archive: {dest.stat().st_size:,} bytes")
        return dest, audit
    for url in URLS:
        try:
            download(url, dest)
            audit.append({"url": url, "status": "DOWNLOADED_VALID_ZIP",
                          "bytes": dest.stat().st_size, "error": ""})
            log(f"Downloaded and validated WWES archive: {dest.stat().st_size:,} bytes")
            return dest, audit
        except Exception as e:
            audit.append({"url": url, "status": "FAILED", "bytes": 0, "error": str(e)})
    return None, audit

def find_file(root: Path, candidates: list[str]) -> Path | None:
    all_files = [p for p in root.rglob("*") if p.is_file() and "__MACOSX" not in p.parts]
    candidate_norm = {norm(x) for x in candidates}
    for p in all_files:
        if norm(p.name) in candidate_norm:
            return p
    return None

def read_delimited(path: Path) -> tuple[list[str], list[dict]]:
    sample = path.read_bytes()[:65536].decode("utf-8-sig", errors="replace")
    delimiter = "\t" if sample.count("\t") >= sample.count(",") else ","
    rows = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fields = reader.fieldnames or []
        for r in reader:
            rows.append(dict(r))
    return fields, rows

def col(headers: list[str], terms: list[str]) -> str | None:
    for h in headers:
        nh = norm(h)
        for t in terms:
            nt = norm(t)
            if nh == nt or nh.endswith("/" + nt) or nh.endswith("#" + nt) or nt in nh:
                return h
    return None

def parse_number(value) -> float | None:
    if value is None:
        return None
    s = str(value).strip().replace(",", ".")
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group())
    except Exception:
        return None

def classify_property(text: str) -> str | None:
    s = norm(text)
    if "bulk modulus" in s or "hydraulic" in s or "turgor" in s:
        return None
    if any(x in s for x in ["modulus of rupture", "mor ", "bending strength", "rupture strength"]):
        return "MOR"
    if any(x in s for x in ["young's modulus", "young modulus",
                            "modulus of elasticity", "elastic modulus", "moe "]):
        return "MOE"
    return None

def normalize_value(prop: str, value, unit: str) -> tuple[float | None, str, str]:
    x = parse_number(value)
    u = norm(unit)
    if x is None:
        return None, "", "NON_NUMERIC"
    if prop == "MOE":
        if "gpa" in u:
            return x * 1000.0, "MPa", "GPa_TO_MPa"
        if "mpa" in u or "n/mm" in u:
            return x, "MPa", "AS_REPORTED"
        return x, unit or "UNKNOWN", "UNIT_UNRESOLVED"
    if prop == "MOR":
        if "gpa" in u:
            return x * 1000.0, "MPa", "GPa_TO_MPa_UNUSUAL"
        if "mpa" in u or "n/mm" in u:
            return x, "MPa", "AS_REPORTED"
        return x, unit or "UNKNOWN", "UNIT_UNRESOLVED"
    return x, unit or "UNKNOWN", "UNIT_UNRESOLVED"

def extract_wwes(archive: Path) -> dict:
    root = WORK / "wwes"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(root)

    meas_path = find_file(root, ["measurements or facts.txt", "measurementorfact.txt",
                                 "measurements_or_facts.txt"])
    occ_path = find_file(root, ["occurrences.txt", "occurrence.txt"])
    taxa_path = find_file(root, ["taxa.txt", "taxon.txt"])
    if not meas_path or not taxa_path:
        raise RuntimeError("Darwin Core archive lacks measurement/taxa tables")

    mh, mr = read_delimited(meas_path)
    th, tr = read_delimited(taxa_path)
    oh, orows = read_delimited(occ_path) if occ_path else ([], [])

    m_occ = col(mh, ["occurrenceID"])
    m_tax = col(mh, ["taxonID"])
    m_type = col(mh, ["measurementType"])
    m_value = col(mh, ["measurementValue"])
    m_unit = col(mh, ["measurementUnit"])
    m_source = col(mh, ["source"])
    m_remarks = col(mh, ["measurementRemarks", "remarks"])
    m_method = col(mh, ["measurementMethod", "method"])

    o_occ = col(oh, ["occurrenceID"])
    o_tax = col(oh, ["taxonID"])
    t_tax = col(th, ["taxonID"])
    t_sci = col(th, ["scientificName"])
    t_canon = col(th, ["canonicalName"])
    t_genus = col(th, ["genus"])
    t_family = col(th, ["family"])

    if not m_type or not m_value:
        raise RuntimeError(f"Cannot identify measurementType/value columns: {mh}")

    occ_to_tax = {}
    if o_occ and o_tax:
        for r in orows:
            occ_to_tax[(r.get(o_occ) or "").strip()] = (r.get(o_tax) or "").strip()

    tax_map = {}
    if t_tax:
        for r in tr:
            tid = (r.get(t_tax) or "").strip()
            tax_map[tid] = {
                "scientific_name": (r.get(t_sci) or "").strip() if t_sci else "",
                "canonical_name": (r.get(t_canon) or "").strip() if t_canon else "",
                "genus": (r.get(t_genus) or "").strip() if t_genus else "",
                "family": (r.get(t_family) or "").strip() if t_family else "",
            }

    inventory = {}
    candidates = []
    for i, r in enumerate(mr, start=1):
        typ = (r.get(m_type) or "").strip()
        remarks = (r.get(m_remarks) or "").strip() if m_remarks else ""
        method = (r.get(m_method) or "").strip() if m_method else ""
        combined = " | ".join([typ, remarks, method])
        inventory[typ] = inventory.get(typ, 0) + 1
        prop = classify_property(combined)
        if not prop:
            continue
        occurrence_id = (r.get(m_occ) or "").strip() if m_occ else ""
        taxon_id = (r.get(m_tax) or "").strip() if m_tax else ""
        if not taxon_id and occurrence_id:
            taxon_id = occ_to_tax.get(occurrence_id, "")
        tx = tax_map.get(taxon_id, {})
        raw_value = (r.get(m_value) or "").strip()
        raw_unit = (r.get(m_unit) or "").strip() if m_unit else ""
        value_mpa, std_unit, conversion = normalize_value(prop, raw_value, raw_unit)
        source = (r.get(m_source) or "").strip() if m_source else ""
        protocol_text = " | ".join([remarks, method])
        protocol_complete = any(x in norm(protocol_text) for x in [
            "longitud", "radial", "tangential", "bending", "astm", "iso", "four-point",
            "three-point", "clear wood", "green", "moisture"
        ])
        candidates.append({
            "candidate_id": f"WWES_{i:07d}",
            "property": prop,
            "value_raw": raw_value,
            "unit_raw": raw_unit,
            "value_standardized": value_mpa if value_mpa is not None else "",
            "unit_standardized": std_unit,
            "conversion_status": conversion,
            "measurement_type_raw": typ,
            "measurement_method_raw": method,
            "measurement_remarks_raw": remarks,
            "protocol_or_orientation_present": "YES" if protocol_complete else "NO",
            "source_reference": source,
            "occurrence_id": occurrence_id,
            "taxon_id": taxon_id,
            "scientific_name": tx.get("scientific_name", ""),
            "canonical_name": tx.get("canonical_name", ""),
            "genus": tx.get("genus", ""),
            "family": tx.get("family", ""),
            "directly_compatible_with_usda": "UNRESOLVED_REQUIRES_PROTOCOL_AUDIT",
        })

    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    fields = list(candidates[0].keys()) if candidates else [
        "candidate_id","property","value_raw","unit_raw","value_standardized",
        "unit_standardized","conversion_status","measurement_type_raw",
        "measurement_method_raw","measurement_remarks_raw",
        "protocol_or_orientation_present","source_reference","occurrence_id",
        "taxon_id","scientific_name","canonical_name","genus","family",
        "directly_compatible_with_usda"
    ]
    with (DATA / "WWES_MECHANICAL_CANDIDATES.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(candidates)

    inv_rows = [{"measurement_type": k, "row_count": v}
                for k, v in sorted(inventory.items(), key=lambda kv: (-kv[1], kv[0]))]
    with (REPORTS / "WWES_MEASUREMENT_TYPE_INVENTORY.csv").open(
            "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["measurement_type","row_count"])
        w.writeheader()
        w.writerows(inv_rows)

    props = {}
    for r in candidates:
        props[r["property"]] = props.get(r["property"], 0) + 1
    taxa = {r["canonical_name"] or r["scientific_name"] or r["taxon_id"]
            for r in candidates if r["canonical_name"] or r["scientific_name"] or r["taxon_id"]}
    sources = {r["source_reference"] for r in candidates if r["source_reference"]}
    numeric = [r for r in candidates if r["value_standardized"] != ""]
    protocol = [r for r in candidates if r["protocol_or_orientation_present"] == "YES"]
    unit_resolved = [r for r in candidates if r["unit_standardized"] == "MPa"]
    metrics = {
        "archive_bytes": archive.stat().st_size,
        "measurement_rows_total": len(mr),
        "taxa_rows_total": len(tr),
        "occurrence_rows_total": len(orows),
        "mechanical_candidates": len(candidates),
        "numeric_mechanical_candidates": len(numeric),
        "property_counts": props,
        "unique_taxa_in_candidates": len(taxa),
        "unique_nonempty_source_references": len(sources),
        "protocol_completeness_fraction": round(len(protocol) / len(candidates), 6) if candidates else 0,
        "unit_mpa_resolved_fraction": round(len(unit_resolved) / len(candidates), 6) if candidates else 0,
        "license_status": "PUBLIC_MIRROR_FOUND_BUT_RECORD_LICENSE_NOT_SPECIFIED",
        "original_dataset_doi": "10.5061/dryad.234",
        "mirror_doi": "10.5281/zenodo.13322441",
    }
    write_json(REPORTS / "WWES_EXTRACTION_METRICS.json", metrics)
    log(
        "WWES extraction: "
        f"mechanical={len(candidates)}, taxa={len(taxa)}, "
        f"properties={props}, protocol_fraction={metrics['protocol_completeness_fraction']}"
    )
    return metrics

def decide(tax_metrics: dict, wwes_metrics: dict | None, access_failed: bool) -> dict:
    if access_failed or not wwes_metrics:
        decision = "YELLOW_NARROW"
        code = "YELLOW_NARROW_WWES_ACCESS_OR_EXTRACTION_NOT_COMPLETED"
        rationale = (
            "Stage0C remains valid, but the candidate compatible-source archive could not "
            "be downloaded or parsed. Full training remains unauthorized."
        )
    else:
        n = int(wwes_metrics.get("numeric_mechanical_candidates", 0))
        taxa = int(wwes_metrics.get("unique_taxa_in_candidates", 0))
        props = wwes_metrics.get("property_counts", {})
        protocol = float(wwes_metrics.get("protocol_completeness_fraction", 0))
        license_ok = wwes_metrics.get("license_status") == "CONFIRMED_REUSABLE"
        has_both = int(props.get("MOE", 0)) > 0 and int(props.get("MOR", 0)) > 0
        if n >= 100 and taxa >= 50 and has_both and protocol >= 0.80 and license_ok:
            decision = "GREEN_FOR_CONTROLLED_PILOT"
            code = "GREEN_COMPATIBLE_SOURCE_CANDIDATE_MEETS_AUTOMATED_GATE"
            rationale = (
                "A substantial mechanical source candidate was identified with both MOE and MOR, "
                "adequate taxonomic breadth, protocol metadata and confirmed reuse status. "
                "Only a controlled pilot and leakage audit are authorized."
            )
        elif n >= 20 and (int(props.get("MOE",0)) > 0 or int(props.get("MOR",0)) > 0):
            decision = "YELLOW_SOURCE_FOUND_PROTOCOL_LICENSE_AUDIT"
            code = "YELLOW_MECHANICAL_SOURCE_FOUND_BUT_COMPATIBILITY_NOT_PROVEN"
            rationale = (
                "The WWES archive contains useful mechanical records, but direct compatibility "
                "with USDA cannot be asserted until protocol/orientation, original-source "
                "independence and reuse terms are audited."
            )
        else:
            decision = "YELLOW_NARROW"
            code = "YELLOW_TOO_FEW_COMPATIBLE_MECHANICAL_RECORDS"
            rationale = (
                "The archive was processed, but the mechanically relevant records are insufficient "
                "for the planned cross-study target."
            )

    return {
        "decision": decision,
        "decision_code": code,
        "rationale": rationale,
        "full_training_authorization": "NO",
        "controlled_pilot_authorization": (
            "YES_AFTER_MANUAL_PROTOCOL_LICENSE_AND_LEAKAGE_AUDIT"
            if decision == "GREEN_FOR_CONTROLLED_PILOT" else "NO"
        ),
        "taxonomy_correction": tax_metrics,
        "wwes_metrics": wwes_metrics or {},
        "allowed_next_steps": [
            "manual protocol and orientation audit of WWES mechanical candidates",
            "resolve taxonomy queue using accepted scientific names and source tables",
            "verify original-source independence and licensing",
            "freeze compatible target regimes",
            "small source-specific diagnostics only after the above checks"
        ],
        "forbidden_next_steps": [
            "pooling WWES with USDA solely because units are both MPa",
            "full HAP-DG training",
            "claiming cross-study validation before source independence is verified",
            "treating GBIF confidence as a match when matchType is NONE"
        ],
        "generated_at": now(),
    }

def write_report(decision: dict, access_audit: list[dict]) -> None:
    taxm = decision["taxonomy_correction"]
    wm = decision.get("wwes_metrics", {})
    report = f"""# WOODPHYS_DG_Q1 — Stage 0D Scientific Gate

## القرار: **{decision['decision']}**

{decision['rationale']}

## تصحيح GBIF
- إجمالي التسميات المفحوصة: {taxm.get('total', 0)}
- Species candidates موثقة آليًا: {taxm.get('verified_species', 0)}
- Genus candidates موثقة آليًا: {taxm.get('verified_genus', 0)}
- أسماء `NONE` أو غير محلولة: {taxm.get('unresolved_none', 0)}
- مراجعة يدوية أخرى: {taxm.get('manual_review', 0)}

رقم Stage 0C البالغ 196 لا يُستخدم كعدد تطابقات صحيحة؛ لأن قيمة
`confidence` لا تعني تطابقًا عندما يكون `matchType=NONE`.

## Worldwide Wood Economics Spectrum
- سجلات القياسات الكلية: {wm.get('measurement_rows_total', 0)}
- المرشح الميكانيكي: {wm.get('mechanical_candidates', 0)}
- المرشح الرقمي: {wm.get('numeric_mechanical_candidates', 0)}
- توزيع الخواص: {json.dumps(wm.get('property_counts', {}), ensure_ascii=False)}
- عدد taxa في المرشحات: {wm.get('unique_taxa_in_candidates', 0)}
- عدد مراجع المصدر غير الفارغة: {wm.get('unique_nonempty_source_references', 0)}
- اكتمال البروتوكول/الاتجاه: {wm.get('protocol_completeness_fraction', 0)}
- حالة الترخيص: {wm.get('license_status', 'NOT_AVAILABLE')}

## الحكم التنفيذي
- Full HAP-DG: **غير مسموح**.
- لا يُدمج WWES مع USDA لمجرد أن الوحدة MPa.
- يجب إثبات اتجاه الاختبار والبروتوكول واستقلال المصدر الأصلي والترخيص.
- المرحلة الناجحة التالية ستكون تجميد مصدر متوافق، ثم baseline تشخيصي صغير.

## سجل الوصول
```json
{json.dumps(access_audit, ensure_ascii=False, indent=2)}
```
"""
    (OUT / "GATE_REPORT_AR.md").write_text(report, encoding="utf-8")

def manifest_and_hashes() -> None:
    files = []
    hashes = []
    for p in sorted(OUT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(OUT).as_posix()
        if rel in {"FILE_MANIFEST.csv", "HASHES_SHA256.csv",
                   "UPLOAD_THIS_STAGE0D_RESULT.zip"}:
            continue
        files.append({"relative_path": rel, "size_bytes": p.stat().st_size})
        hashes.append({"relative_path": rel, "sha256": sha256(p)})
    with (OUT / "FILE_MANIFEST.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["relative_path","size_bytes"])
        w.writeheader(); w.writerows(files)
    with (OUT / "HASHES_SHA256.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["relative_path","sha256"])
        w.writeheader(); w.writerows(hashes)

def make_upload_zip() -> Path:
    path = OUT / "UPLOAD_THIS_STAGE0D_RESULT.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and p != path and "_work" not in p.parts:
                z.write(p, p.relative_to(OUT))
    return path

def main() -> int:
    started = now()
    if OUT.exists():
        shutil.rmtree(OUT)
    for p in [OUT, WORK, RAW, DATA, REPORTS, TAX]:
        p.mkdir(parents=True, exist_ok=True)

    status = {
        "stage": "STAGE0D", "version": "1.0.0",
        "status": "running", "started_at": started
    }
    write_json(OUT / "STATUS.json", status)

    access_audit = []
    try:
        stage0c = validate_stage0c()
        tax_metrics = taxonomy_correction(stage0c)
        archive, access_audit = acquire_wwes()
        with (REPORTS / "SOURCE_ACCESS_AUDIT.csv").open(
                "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["url","status","bytes","error"])
            w.writeheader(); w.writerows(access_audit)

        wwes_metrics = None
        access_failed = archive is None
        if archive is not None:
            wwes_metrics = extract_wwes(archive)

        decision = decide(tax_metrics, wwes_metrics, access_failed)
        write_json(OUT / "SCIENTIFIC_GATE_DECISION.json", decision)
        write_report(decision, access_audit)

        required = [
            "STATUS.json", "SCIENTIFIC_GATE_DECISION.json", "GATE_REPORT_AR.md",
            "taxonomy/GBIF_TAXONOMY_CORRECTED_AUDIT.csv",
            "taxonomy/TAXONOMY_CORRECTION_SUMMARY.json",
            "reports/SOURCE_ACCESS_AUDIT.csv",
        ]
        missing = [x for x in required if not (OUT / x).exists()]
        verification = {
            "required_files_missing": missing,
            "stage0c_verified": True,
            "taxonomy_none_not_counted_as_match":
                tax_metrics.get("unresolved_none", 0) > 0,
            "wwes_archive_validated": bool(archive and zipfile.is_zipfile(archive)),
            "full_training_prevented":
                decision.get("full_training_authorization") == "NO",
            "verification_passed": len(missing) == 0,
        }
        write_json(OUT / "OUTPUT_VERIFICATION.json", verification)
        write_json(OUT / "CHECKPOINT.json", {
            "completed_steps": [
                "stage0c_verification", "taxonomy_metric_correction",
                "wwes_acquisition_attempt", "mechanical_candidate_extraction",
                "scientific_gate"
            ],
            "resume_supported": True, "timestamp": now()
        })
        status.update({
            "status": "completed",
            "completed_at": now(),
            "scientific_decision": decision["decision"]
        })
        write_json(OUT / "STATUS.json", status)
        write_json(OUT / "SUCCESS_MARKER.json", {
            "success": True, "stage": "STAGE0D", "version": "1.0.0",
            "scientific_decision": decision["decision"], "timestamp": now()
        })
        (OUT / "RUN_LOG.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
        (OUT / "ERROR_LOG.txt").write_text("", encoding="utf-8")
        manifest_and_hashes()
        make_upload_zip()
        log(f"Completed. Scientific decision: {decision['decision']}")
        (OUT / "RUN_LOG.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
        return 0
    except Exception as e:
        ERROR_LINES.append(f"[{now()}] {type(e).__name__}: {e}")
        status.update({"status": "failed", "completed_at": now(), "error": str(e)})
        write_json(OUT / "STATUS.json", status)
        write_json(OUT / "SUCCESS_MARKER.json", {
            "success": False, "stage": "STAGE0D",
            "version": "1.0.0", "timestamp": now(), "error": str(e)
        })
        (OUT / "RUN_LOG.txt").write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
        (OUT / "ERROR_LOG.txt").write_text("\n".join(ERROR_LINES) + "\n", encoding="utf-8")
        try:
            manifest_and_hashes()
            make_upload_zip()
        except Exception:
            pass
        print(ERROR_LINES[-1], file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
