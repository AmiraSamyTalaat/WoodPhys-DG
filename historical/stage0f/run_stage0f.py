
from __future__ import annotations
import csv, json, hashlib, shutil, zipfile, datetime as dt, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"stage0f_output"
CAN=ROOT/"canonical"
LOG=[]

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def log(x):
    s=f"[{now()}] {x}"; LOG.append(s); print(s,flush=True)
def write_json(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding="utf-8")
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def read_csv(p):
    with p.open("r",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))
def stable_fold(text,k=5):
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:8],16)%k

def main():
    if OUT.exists(): shutil.rmtree(OUT)
    for d in [OUT,OUT/"freeze",OUT/"splits",OUT/"reports"]: d.mkdir(parents=True,exist_ok=True)
    write_json(OUT/"STATUS.json",{"stage":"STAGE0F","status":"running","started_at":now()})
    try:
        files={
            "USDA":CAN/"usda_clearwood_primary_model_ready.csv",
            "JAPANESE_LARCH":CAN/"japanese_larch_overall_model_ready.csv",
            "JAPANESE_LARCH_RADIAL_SECONDARY":CAN/"japanese_larch_radial_repeated_measures_secondary.csv",
            "SPRUCE":CAN/"spruce_structural_board_separate_regime.csv",
            "PINUS_RADIAL":CAN/"pinus_nigra_radial_separate_regime.csv",
        }
        missing=[k for k,p in files.items() if not p.exists()]
        if missing: raise RuntimeError(f"Missing canonical files: {missing}")
        rows={k:read_csv(p) for k,p in files.items()}
        expected={"USDA":422,"JAPANESE_LARCH":488,"SPRUCE":2522}
        checks={}
        for k,n in expected.items():
            checks[k]={"actual_rows":len(rows[k]),"expected_rows":n,"pass":len(rows[k])==n}
        if not all(x["pass"] for x in checks.values()):
            raise RuntimeError(f"Canonical row checks failed: {checks}")

        # Copy immutable freeze files.
        frozen={}
        for k,p in files.items():
            dest=OUT/"freeze"/p.name
            shutil.copy2(p,dest)
            frozen[k]={"file":dest.relative_to(OUT).as_posix(),"sha256":sha(dest),"rows":len(rows[k])}

        # USDA domain-shift manifest.
        usda_manifest=[]
        for r in rows["USDA"]:
            table=r.get("table_id","")
            if "5_3A" in table: role="DEVELOPMENT_US"
            elif "5_4A" in table: role="CALIBRATION_CANADA"
            elif "5_5A" in table: role="EXTERNAL_OTHER_IMPORTS"
            else: role="UNASSIGNED"
            genus=r.get("genus") or "UNKNOWN"
            rr={"record_id":r.get("record_id",""),"domain_role":role,
                "genus_group_fold_5":stable_fold(genus,5),
                "genus":genus,"species_label_raw":r.get("species_label_raw",""),
                "moisture_state":r.get("moisture_state","")}
            usda_manifest.append(rr)
        with (OUT/"splits"/"USDA_DOMAIN_AND_TAXON_SPLIT_MANIFEST.csv").open(
            "w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(usda_manifest[0]));w.writeheader();w.writerows(usda_manifest)

        # Japanese larch: site 431 development, site 496 external; family grouping retained.
        jl_manifest=[]
        for r in rows["JAPANESE_LARCH"]:
            site=str(r.get("TestID","")).split(".")[0]
            role="DEVELOPMENT_SITE_431" if site=="431" else (
                "EXTERNAL_SITE_496" if site=="496" else "UNASSIGNED")
            fam=r.get("Family_code") or "UNKNOWN"
            jl_manifest.append({
                "record_id":r.get("record_id",""),
                "TreeID":r.get("TreeID",""),
                "TestID":site,
                "Family_code":fam,
                "site_role":role,
                "family_group_fold_5":stable_fold(fam,5)
            })
        with (OUT/"splits"/"JAPANESE_LARCH_SITE_FAMILY_SPLIT_MANIFEST.csv").open(
            "w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(jl_manifest[0]));w.writeheader();w.writerows(jl_manifest)

        # Spruce: preserve source-provided split only.
        sp_manifest=[{
            "record_id":r.get("record_id",""),
            "sample_name":r.get("sample_name",""),
            "provided_split":r.get("provided_split",""),
            "board_identity_preserved":"YES"
        } for r in rows["SPRUCE"]]
        with (OUT/"splits"/"SPRUCE_PROVIDED_SPLIT_MANIFEST.csv").open(
            "w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(sp_manifest[0]));w.writeheader();w.writerows(sp_manifest)

        regime_matrix=[
            {"regime":"USDA_CLEARWOOD","rows":len(rows["USDA"]),"level":"species means",
             "primary_shift":"geography/table + held-out genus","direct_pooling":"NO"},
            {"regime":"JAPANESE_LARCH","rows":len(rows["JAPANESE_LARCH"]),"level":"individual tree",
             "primary_shift":"test site + family","direct_pooling":"NO"},
            {"regime":"SPRUCE_STRUCTURAL","rows":len(rows["SPRUCE"]),"level":"structural board",
             "primary_shift":"provided train/test","direct_pooling":"NO"},
            {"regime":"PINUS_NIGRA_RADIAL","rows":len(rows["PINUS_RADIAL"]),"level":"individual tree radial",
             "primary_shift":"site/radial context","direct_pooling":"NO"},
        ]
        with (OUT/"reports"/"REGIME_COMPATIBILITY_MATRIX.csv").open(
            "w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(regime_matrix[0]));w.writeheader();w.writerows(regime_matrix)

        decision={
            "decision":"GREEN_FOR_REGIME_SPECIFIC_DIAGNOSTIC_PILOT",
            "full_hap_dg_authorization":"NO",
            "direct_cross_regime_pooling":"FORBIDDEN",
            "broad_cross_study_cross_species_claim":"FORBIDDEN",
            "authorized_next_stage":"STAGE1_DIAGNOSTIC_BASELINES_AND_SHIFT_STRESS_TESTS",
            "authorized_scope":[
                "separate baseline models per mechanical regime",
                "USDA geographic/table and held-out-genus diagnostics",
                "Japanese larch held-out-site and held-out-family diagnostics",
                "Spruce source-provided train/test diagnostics",
                "source-specific conformal and shift-score feasibility only"
            ],
            "freeze_files":frozen,
            "row_checks":checks,
            "generated_at":now()
        }
        write_json(OUT/"SCIENTIFIC_GATE_DECISION.json",decision)
        report=f"""# WOODPHYS_DG_Q1 — Stage 0F

## القرار
**GREEN_FOR_REGIME_SPECIFIC_DIAGNOSTIC_PILOT**

هذا لا يعني السماح بالمنهج الكامل أو دمج المصادر.

## البيانات المجمدة
- USDA clear wood: **{len(rows['USDA'])}**
- Japanese larch overall: **{len(rows['JAPANESE_LARCH'])}**
- Spruce structural boards: **{len(rows['SPRUCE'])}**
- Pinus nigra radial: **{len(rows['PINUS_RADIAL'])}**
- Japanese larch radial repeated measures: **{len(rows['JAPANESE_LARCH_RADIAL_SECONDARY'])}**، ثانوي فقط.

## المسموح
- Baselines مستقلة لكل regime.
- اختبارات held-out genus وgeographic table داخل USDA.
- held-out site وfamily داخل Japanese larch.
- shift score وconformal بصورة تشخيصية مصدرية.

## الممنوع
- دمج الأنظمة الأربعة في Regression واحد.
- ادعاء broad cross-study external validation.
- Full HAP-DG.
- استخدام repeated radial rows كعينات مستقلة.
"""
        (OUT/"GATE_REPORT_AR.md").write_text(report,encoding="utf-8")
        write_json(OUT/"OUTPUT_VERIFICATION.json",{
            "verification_passed":True,"canonical_row_checks":checks,
            "direct_pooling_prevented":True,
            "full_training_prevented":True,
            "split_manifests_created":3
        })
        write_json(OUT/"SUCCESS_MARKER.json",{
            "success":True,"stage":"STAGE0F","version":"1.0.0",
            "decision":decision["decision"],"timestamp":now()
        })
        write_json(OUT/"STATUS.json",{
            "stage":"STAGE0F","version":"1.0.0","status":"completed",
            "decision":decision["decision"],"completed_at":now()
        })
        (OUT/"RUN_LOG.txt").write_text("\n".join(LOG)+"\n",encoding="utf-8")
        (OUT/"ERROR_LOG.txt").write_text("",encoding="utf-8")

        manifest=[]; hashes=[]
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and p.name not in {"FILE_MANIFEST.csv","HASHES_SHA256.csv",
                                             "UPLOAD_THIS_STAGE0F_RESULT.zip"}:
                rel=p.relative_to(OUT).as_posix()
                manifest.append({"relative_path":rel,"size_bytes":p.stat().st_size})
                hashes.append({"relative_path":rel,"sha256":sha(p)})
        with (OUT/"FILE_MANIFEST.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["relative_path","size_bytes"]);w.writeheader();w.writerows(manifest)
        with (OUT/"HASHES_SHA256.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["relative_path","sha256"]);w.writeheader();w.writerows(hashes)
        upload=OUT/"UPLOAD_THIS_STAGE0F_RESULT.zip"
        with zipfile.ZipFile(upload,"w",zipfile.ZIP_DEFLATED) as z:
            for p in sorted(OUT.rglob("*")):
                if p.is_file() and p!=upload:
                    z.write(p,p.relative_to(OUT))
        log("Completed. Decision: GREEN_FOR_REGIME_SPECIFIC_DIAGNOSTIC_PILOT")
        (OUT/"RUN_LOG.txt").write_text("\n".join(LOG)+"\n",encoding="utf-8")
        return 0
    except Exception as e:
        write_json(OUT/"STATUS.json",{"stage":"STAGE0F","status":"failed","error":str(e),"time":now()})
        write_json(OUT/"SUCCESS_MARKER.json",{"success":False,"error":str(e),"time":now()})
        (OUT/"RUN_LOG.txt").write_text("\n".join(LOG)+"\n",encoding="utf-8")
        (OUT/"ERROR_LOG.txt").write_text(f"{type(e).__name__}: {e}\n",encoding="utf-8")
        print(f"{type(e).__name__}: {e}",file=sys.stderr)
        return 1
if __name__=="__main__":
    raise SystemExit(main())
