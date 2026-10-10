
from __future__ import annotations
import csv, datetime as dt, hashlib, io, json, os, re, shutil, sys, time
import urllib.request, urllib.error, zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "stage0e_output"
WORK = OUT / "_work"
RAW = OUT / "raw"
DATA = OUT / "data"
REPORTS = OUT / "reports"
INPUT0C = ROOT / "input" / "STAGE0C_VERIFIED_INPUT.zip"
AUDIT0D = ROOT / "embedded_audit" / "STAGE0D_CORRECTED_SCIENTIFIC_AUDIT.json"
LOG, ERR = [], []

FIGSHARE_IDS = {
    "FIGSHARE_JAPANESE_LARCH_21397674": 21397674,
    "FIGSHARE_EUCALYPTUS_14321961": 14321961,
    "RS_FIGSHARE_SUPPLEMENT_25449816": 25449816,
}
ZENODO_STATIC_URLS = [
    "https://zenodo.org/records/10218710/files/3-NA_Static_Bend.csv?download=1",
    "https://zenodo.org/record/10218710/files/3-NA_Static_Bend.csv?download=1",
]

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def log(msg):
    line=f"[{now()}] {msg}"
    LOG.append(line); print(line, flush=True)

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def norm(s):
    s=str(s or "").strip().lower()
    s=re.sub(r"[^a-z0-9%]+", "", s)
    return s

def fetch_bytes(url, attempts=4, max_mb=25):
    headers={
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/148 Safari/537.36",
        "Accept":"*/*",
    }
    last=None
    for i in range(1, attempts+1):
        try:
            req=urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                length=r.headers.get("Content-Length")
                if length and int(length) > max_mb*1024*1024:
                    raise RuntimeError(f"File exceeds {max_mb} MB")
                data=r.read(max_mb*1024*1024+1)
            if len(data)>max_mb*1024*1024:
                raise RuntimeError(f"File exceeds {max_mb} MB")
            if len(data)<20:
                raise RuntimeError(f"Downloaded object too small: {len(data)}")
            low=data[:500].lower()
            if b"<html" in low or b"captcha" in low or b"access denied" in low:
                raise RuntimeError("HTML/protection page returned instead of data")
            return data
        except Exception as e:
            last=e
            log(f"Download attempt {i}/{attempts} failed: {url} :: {e}")
            time.sleep(i*2)
    raise RuntimeError(f"All attempts failed: {last}")

def verify_stage0c():
    if not INPUT0C.exists(): raise FileNotFoundError(INPUT0C)
    dst=WORK/"stage0c"; dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(INPUT0C) as z: z.extractall(dst)
    success=json.loads((dst/"SUCCESS_MARKER.json").read_text(encoding="utf-8"))
    verify=json.loads((dst/"OUTPUT_VERIFICATION.json").read_text(encoding="utf-8"))
    if not success.get("success") or not verify.get("verification_passed"):
        raise RuntimeError("Stage0C input is not verified")
    usda=dst/"data"/"usda_clearwood_longitudinal.csv"
    if not usda.exists(): raise RuntimeError("USDA canonical table missing")
    log("Verified Stage0C and USDA canonical table")
    return dst, usda

def read_csv_any(path):
    data=path.read_bytes()
    text=None
    for enc in ("utf-8-sig","utf-8","latin-1"):
        try:
            text=data.decode(enc); break
        except Exception: pass
    if text is None: text=data.decode("utf-8", errors="replace")
    sample=text[:10000]
    delim="\t" if sample.count("\t")>sample.count(",") else ","
    return list(csv.DictReader(io.StringIO(text), delimiter=delim))

def xlsx_rows(path):
    # Minimal stdlib XLSX reader: first visible worksheet, shared strings supported.
    with zipfile.ZipFile(path) as z:
        names=set(z.namelist())
        shared=[]
        if "xl/sharedStrings.xml" in names:
            root=ET.fromstring(z.read("xl/sharedStrings.xml"))
            ns={"m":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in root.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//m:t", ns)))
        wb=ET.fromstring(z.read("xl/workbook.xml"))
        ns={"m":"http://schemas.openxmlformats.org/spreadsheetml/2006/main",
            "r":"http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
        sheet=wb.find("m:sheets/m:sheet", ns)
        rid=sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        rel=ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target=None
        for r in rel:
            if r.attrib.get("Id")==rid: target=r.attrib["Target"]; break
        target=target.lstrip("/")
        if not target.startswith("xl/"): target="xl/"+target
        root=ET.fromstring(z.read(target))
        rows=[]
        for rr in root.findall(".//m:sheetData/m:row", ns):
            vals={}
            for c in rr.findall("m:c",ns):
                ref=c.attrib.get("r","")
                col=re.match(r"[A-Z]+",ref).group(0)
                typ=c.attrib.get("t")
                v=c.find("m:v",ns)
                val="" if v is None else (v.text or "")
                if typ=="s" and val:
                    val=shared[int(val)]
                elif typ=="inlineStr":
                    val="".join(t.text or "" for t in c.findall(".//m:t",ns))
                vals[col]=val
            rows.append(vals)
    if not rows: return []
    def colnum(s):
        n=0
        for ch in s: n=n*26+ord(ch)-64
        return n
    cols=sorted({k for r in rows for k in r}, key=colnum)
    matrix=[[r.get(c,"") for c in cols] for r in rows]
    # Find a likely header row.
    hi=0
    for i,row in enumerate(matrix[:30]):
        joined=" ".join(str(x).lower() for x in row)
        if ("moe" in joined or "modulus of elasticity" in joined) and ("mor" in joined or "modulus of rupture" in joined):
            hi=i; break
    headers=[str(x).strip() or f"COL_{j+1}" for j,x in enumerate(matrix[hi])]
    out=[]
    for row in matrix[hi+1:]:
        if any(str(x).strip() for x in row):
            out.append({headers[j]:row[j] if j<len(row) else "" for j in range(len(headers))})
    return out

def parse_tabular(path):
    suf=path.suffix.lower()
    if suf in (".csv",".tsv",".txt"): return read_csv_any(path)
    if suf==".xlsx" and zipfile.is_zipfile(path): return xlsx_rows(path)
    return []

def detect_columns(rows):
    if not rows: return {}
    headers=list(rows[0].keys())
    def pick(terms):
        for h in headers:
            nh=norm(h)
            for t in terms:
                if norm(t)==nh or norm(t) in nh: return h
        return None
    return {
        "species":pick(["scientific name","species","wood species","tree species","common name"]),
        "moe":pick(["moe","modulus of elasticity","static modulus"]),
        "mor":pick(["mor","modulus of rupture","bending strength","flexural strength"]),
        "density":pick(["density","specific gravity","air dry density","sgbasic"]),
        "moisture":pick(["moisture","mc"]),
        "family":pick(["family"]),
        "genus":pick(["genus"]),
        "source":pick(["source","reference","citation"]),
        "sample_id":pick(["sample","specimen","tree","id"]),
    }

def num(v):
    s=str(v or "").strip().replace(",",".")
    m=re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",s)
    if not m: return None
    try:return float(m.group())
    except:return None

def normalize_moe(v, header):
    x=num(v)
    if x is None:return None
    h=str(header or "").lower()
    if "gpa" in h or (x<100 and x>1): return x*1000
    return x

def load_usda(path):
    rows=read_csv_any(path)
    tuples=set()
    species=set()
    for r in rows:
        mor=num(r.get("mor_mpa")); moe=num(r.get("moe_mpa"))
        if mor is not None and moe is not None:
            tuples.add((round(mor,1),round(moe,0)))
        species.add((r.get("species_label_raw") or "").strip().lower())
    return rows, tuples, species

def figshare_metadata(article_id):
    url=f"https://api.figshare.com/v2/articles/{article_id}"
    data=fetch_bytes(url, max_mb=5)
    return json.loads(data.decode("utf-8"))

def save_and_parse(source_id, name, data):
    safe=re.sub(r"[^A-Za-z0-9._-]+","_",name)
    p=RAW/source_id/safe
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    rows=parse_tabular(p)
    return p, rows

def audit_rows(source_id, rows, usda_tuples, declared_license, article_title="", file_name=""):
    cols=detect_columns(rows)
    extracted=[]
    exact_pairs=0
    numeric_pairs=0
    for i,r in enumerate(rows, start=1):
        moe=normalize_moe(r.get(cols["moe"]) if cols.get("moe") else None, cols.get("moe"))
        mor=num(r.get(cols["mor"]) if cols.get("mor") else None)
        if moe is not None and mor is not None:
            numeric_pairs+=1
            if (round(mor,1),round(moe,0)) in usda_tuples: exact_pairs+=1
        extracted.append({
            "source_id":source_id,
            "source_title":article_title,
            "file_name":file_name,
            "row_number":i,
            "species_raw":r.get(cols["species"],"") if cols.get("species") else "",
            "family_raw":r.get(cols["family"],"") if cols.get("family") else "",
            "genus_raw":r.get(cols["genus"],"") if cols.get("genus") else "",
            "moe_mpa_candidate":"" if moe is None else moe,
            "mor_mpa_candidate":"" if mor is None else mor,
            "density_candidate":r.get(cols["density"],"") if cols.get("density") else "",
            "moisture_candidate":r.get(cols["moisture"],"") if cols.get("moisture") else "",
            "sample_id_candidate":r.get(cols["sample_id"],"") if cols.get("sample_id") else "",
            "source_reference_candidate":r.get(cols["source"],"") if cols.get("source") else "",
            "declared_license":declared_license,
        })
    overlap=exact_pairs/numeric_pairs if numeric_pairs else 0
    if numeric_pairs and overlap>=0.70:
        independence="LIKELY_DERIVED_FROM_OR_DUPLICATE_OF_USDA"
    elif numeric_pairs:
        independence="NO_EXACT_VALUE_DUPLICATION_SIGNAL_DETECTED"
    else:
        independence="NO_USABLE_MOE_MOR_PAIR"
    return extracted, {
        "source_id":source_id,
        "source_title":article_title,
        "file_name":file_name,
        "rows_parsed":len(rows),
        "numeric_moe_mor_pairs":numeric_pairs,
        "exact_usda_pair_matches":exact_pairs,
        "exact_pair_overlap_fraction":round(overlap,6),
        "independence_screen":independence,
        "detected_columns":cols,
        "declared_license":declared_license,
    }

def main():
    if OUT.exists(): shutil.rmtree(OUT)
    for d in (OUT,WORK,RAW,DATA,REPORTS): d.mkdir(parents=True, exist_ok=True)
    write_json(OUT/"STATUS.json",{
        "stage":"STAGE0E","version":"1.0.0","status":"running","started_at":now()
    })
    try:
        stage0c, usda_path=verify_stage0c()
        usda_rows, usda_tuples, usda_species=load_usda(usda_path)
        corrected0d=json.loads(AUDIT0D.read_text(encoding="utf-8"))
        write_json(REPORTS/"STAGE0D_CORRECTED_AUDIT_COPY.json", corrected0d)

        access=[]
        summaries=[]
        all_extract=[]
        source_success={}

        # Zenodo static bending CSV.
        z_id="ZENODO_STATIC_BEND_10218710"
        z_ok=False
        for url in ZENODO_STATIC_URLS:
            try:
                data=fetch_bytes(url,max_mb=5)
                p,rows=save_and_parse(z_id,"3-NA_Static_Bend.csv",data)
                ext,summ=audit_rows(z_id,rows,usda_tuples,"CC-BY-4.0",
                    "Associations between shade tolerance and wood specific gravity",p.name)
                all_extract.extend(ext); summaries.append(summ)
                access.append({"source_id":z_id,"url":url,"status":"DOWNLOADED","bytes":len(data),"error":""})
                z_ok=True; break
            except Exception as e:
                access.append({"source_id":z_id,"url":url,"status":"FAILED","bytes":0,"error":str(e)})
        source_success[z_id]=z_ok

        # Figshare records.
        for sid,aid in FIGSHARE_IDS.items():
            try:
                meta=figshare_metadata(aid)
                title=meta.get("title","")
                lic=(meta.get("license") or {}).get("name","")
                files=meta.get("files",[])
                candidate_files=[]
                for f in files:
                    name=f.get("name","")
                    size=int(f.get("size") or 0)
                    if size<=25*1024*1024 and Path(name).suffix.lower() in {".csv",".tsv",".txt",".xlsx"}:
                        candidate_files.append(f)
                if not candidate_files:
                    summaries.append({
                        "source_id":sid,"source_title":title,"file_name":"",
                        "rows_parsed":0,"numeric_moe_mor_pairs":0,
                        "exact_usda_pair_matches":0,"exact_pair_overlap_fraction":0,
                        "independence_screen":"NO_SMALL_TABULAR_FILE_FOUND",
                        "detected_columns":{},"declared_license":lic
                    })
                    access.append({"source_id":sid,"url":f"https://api.figshare.com/v2/articles/{aid}",
                                   "status":"METADATA_ONLY_NO_SMALL_TABULAR_FILE","bytes":0,"error":""})
                    source_success[sid]=False
                    continue
                any_ok=False
                for f in candidate_files:
                    try:
                        data=fetch_bytes(f["download_url"],max_mb=25)
                        p,rows=save_and_parse(sid,f["name"],data)
                        ext,summ=audit_rows(sid,rows,usda_tuples,lic,title,p.name)
                        all_extract.extend(ext); summaries.append(summ)
                        access.append({"source_id":sid,"url":f["download_url"],"status":"DOWNLOADED",
                                       "bytes":len(data),"error":""})
                        any_ok=True
                    except Exception as e:
                        access.append({"source_id":sid,"url":f.get("download_url",""),"status":"FAILED",
                                       "bytes":0,"error":str(e)})
                source_success[sid]=any_ok
            except Exception as e:
                access.append({"source_id":sid,"url":f"https://api.figshare.com/v2/articles/{aid}",
                               "status":"FAILED","bytes":0,"error":str(e)})
                source_success[sid]=False

        # Write files.
        with (REPORTS/"SOURCE_ACCESS_AUDIT.csv").open("w",encoding="utf-8-sig",newline="") as f:
            fields=["source_id","url","status","bytes","error"]
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(access)
        sum_fields=["source_id","source_title","file_name","rows_parsed","numeric_moe_mor_pairs",
                    "exact_usda_pair_matches","exact_pair_overlap_fraction","independence_screen",
                    "detected_columns","declared_license"]
        with (REPORTS/"SOURCE_COMPATIBILITY_AND_INDEPENDENCE_AUDIT.csv").open(
            "w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=sum_fields);w.writeheader()
            for s in summaries:
                ss=dict(s);ss["detected_columns"]=json.dumps(ss.get("detected_columns",{}),ensure_ascii=False)
                w.writerow(ss)
        ext_fields=["source_id","source_title","file_name","row_number","species_raw","family_raw",
                    "genus_raw","moe_mpa_candidate","mor_mpa_candidate","density_candidate",
                    "moisture_candidate","sample_id_candidate","source_reference_candidate","declared_license"]
        with (DATA/"EXTERNAL_BENDING_CANDIDATES.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=ext_fields);w.writeheader();w.writerows(all_extract)

        independent=[s for s in summaries if s.get("numeric_moe_mor_pairs",0)>=10
                     and s.get("independence_screen")=="NO_EXACT_VALUE_DUPLICATION_SIGNAL_DETECTED"]
        derived=[s for s in summaries if s.get("independence_screen")=="LIKELY_DERIVED_FROM_OR_DUPLICATE_OF_USDA"]
        total_ind_rows=sum(int(s.get("numeric_moe_mor_pairs",0)) for s in independent)

        # Conservative gate: two independent sources and broad enough rows still only authorize controlled pilot.
        if len(independent)>=2 and total_ind_rows>=100:
            decision="GREEN_FOR_CONTROLLED_FREEZE_AND_PILOT"
            rationale=("At least two non-duplicated external bending sources with usable MOE/MOR pairs "
                       "were identified. Regime and measurement-level harmonization remains required.")
            pilot="YES_AFTER_MANUAL_ROW_AND_PROTOCOL_AUDIT"
        elif len(independent)>=1:
            decision="YELLOW_EXTERNAL_VALIDATION_ONLY"
            rationale=("At least one potentially independent MOE/MOR source was found, but the evidence "
                       "is insufficient for full cross-study domain generalization.")
            pilot="NO"
        else:
            decision="YELLOW_NARROW_USDA_PRIMARY"
            rationale=("No sufficiently independent compatible pair of external bending sources was "
                       "verified. USDA remains the only broad cross-species longitudinal source.")
            pilot="NO"

        dec={
            "decision":decision,"rationale":rationale,
            "full_training_authorization":"NO",
            "controlled_pilot_authorization":pilot,
            "independent_usable_sources":len(independent),
            "independent_usable_rows":total_ind_rows,
            "likely_usda_derived_sources":len(derived),
            "source_summaries":summaries,
            "allowed_next_steps":[
                "manual protocol and measurement-level audit",
                "taxonomy normalization of independent candidate rows",
                "freeze source-specific datasets without direct pooling",
                "redesign title and claims if only USDA remains broad"
            ],
            "forbidden_next_steps":[
                "counting derived USDA values as external validation",
                "direct pooling of species means and specimen-level rows",
                "full HAP-DG training",
                "claiming cross-study generalization without independent domains"
            ],
            "generated_at":now()
        }
        write_json(OUT/"SCIENTIFIC_GATE_DECISION.json",dec)

        report=f"""# WOODPHYS_DG_Q1 — Stage 0E

## القرار: **{decision}**

{rationale}

## تصحيح Stage 0D
- WWES mirror measurements: **{corrected0d['archive_validation']['measurement_rows']:,}**
- MOE rows: **0**
- MOR rows: **0**
- النتيجة: مصدر كثافة فقط، ولا يُحسب مصدرًا ميكانيكيًا.

## نتائج المصادر
- مصادر مستقلة قابلة للاستخدام مبدئيًا: **{len(independent)}**
- صفوف MOE/MOR المستقلة المرشحة: **{total_ind_rows}**
- مصادر تبدو مشتقة/مكررة من USDA: **{len(derived)}**

## الحكم التنفيذي
- Full HAP-DG: **غير مسموح**.
- أي قرار أخضر هنا يسمح فقط بتجميد مضبوط وpilot صغير بعد مراجعة الصفوف.
- افتح `reports/SOURCE_COMPATIBILITY_AND_INDEPENDENCE_AUDIT.csv`.
"""
        (OUT/"GATE_REPORT_AR.md").write_text(report,encoding="utf-8")

        required=["STATUS.json","SCIENTIFIC_GATE_DECISION.json","GATE_REPORT_AR.md",
                  "reports/SOURCE_ACCESS_AUDIT.csv",
                  "reports/SOURCE_COMPATIBILITY_AND_INDEPENDENCE_AUDIT.csv",
                  "data/EXTERNAL_BENDING_CANDIDATES.csv"]
        missing=[x for x in required if not (OUT/x).exists()]
        write_json(OUT/"OUTPUT_VERIFICATION.json",{
            "missing_required_files":missing,
            "stage0c_verified":True,
            "stage0d_corrected_audit_embedded":True,
            "wwes_not_counted_as_mechanical":True,
            "usda_duplicate_screen_executed":True,
            "full_training_prevented":True,
            "verification_passed":not missing
        })
        write_json(OUT/"CHECKPOINT.json",{
            "completed_steps":["stage0c_verify","stage0d_corrected_audit",
                               "source_acquisition","tabular_detection","usda_overlap_screen",
                               "scientific_gate"],
            "timestamp":now()
        })
        write_json(OUT/"STATUS.json",{
            "stage":"STAGE0E","version":"1.0.0","status":"completed",
            "scientific_decision":decision,"completed_at":now()
        })
        write_json(OUT/"SUCCESS_MARKER.json",{
            "success":True,"stage":"STAGE0E","version":"1.0.0",
            "scientific_decision":decision,"timestamp":now()
        })
        (OUT/"RUN_LOG.txt").write_text("\n".join(LOG)+"\n",encoding="utf-8")
        (OUT/"ERROR_LOG.txt").write_text("",encoding="utf-8")

        # Manifest and hashes.
        manifest=[]; hashes=[]
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and p.name not in {"FILE_MANIFEST.csv","HASHES_SHA256.csv",
                                             "UPLOAD_THIS_STAGE0E_RESULT.zip"} and "_work" not in p.parts:
                rel=p.relative_to(OUT).as_posix()
                manifest.append({"relative_path":rel,"size_bytes":p.stat().st_size})
                hashes.append({"relative_path":rel,"sha256":sha256(p)})
        with (OUT/"FILE_MANIFEST.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["relative_path","size_bytes"]);w.writeheader();w.writerows(manifest)
        with (OUT/"HASHES_SHA256.csv").open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["relative_path","sha256"]);w.writeheader();w.writerows(hashes)
        upload=OUT/"UPLOAD_THIS_STAGE0E_RESULT.zip"
        with zipfile.ZipFile(upload,"w",zipfile.ZIP_DEFLATED) as z:
            for p in sorted(OUT.rglob("*")):
                if p.is_file() and p!=upload and "_work" not in p.parts:
                    z.write(p,p.relative_to(OUT))
        log(f"Completed. Scientific decision: {decision}")
        (OUT/"RUN_LOG.txt").write_text("\n".join(LOG)+"\n",encoding="utf-8")
        return 0
    except Exception as e:
        ERR.append(f"[{now()}] {type(e).__name__}: {e}")
        write_json(OUT/"STATUS.json",{
            "stage":"STAGE0E","version":"1.0.0","status":"failed",
            "error":str(e),"completed_at":now()
        })
        write_json(OUT/"SUCCESS_MARKER.json",{
            "success":False,"stage":"STAGE0E","version":"1.0.0",
            "error":str(e),"timestamp":now()
        })
        (OUT/"RUN_LOG.txt").write_text("\n".join(LOG)+"\n",encoding="utf-8")
        (OUT/"ERROR_LOG.txt").write_text("\n".join(ERR)+"\n",encoding="utf-8")
        print(ERR[-1],file=sys.stderr)
        return 1

if __name__=="__main__":
    raise SystemExit(main())
