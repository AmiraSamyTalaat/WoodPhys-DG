from __future__ import annotations
from pathlib import Path
import csv, json, zipfile, hashlib, shutil, urllib.request, urllib.parse, time, traceback, statistics, math, os
from datetime import datetime, timezone

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'stage0c_output'
INPUT_ZIP=BASE/'inputs'/'STAGE0B_VERIFIED_INPUT.zip'
SPRUCE=BASE/'inputs'/'dataset_spruce.csv'
VERSION='1.0.0'

def now(): return datetime.now(timezone.utc).isoformat()
def write_json(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def log(msg):
    s=f'[{now()}] {msg}'
    print(s,flush=True)
    with open(OUT/'RUN_LOG.txt','a',encoding='utf-8') as f:f.write(s+'\n')
def fnum(v):
    try:
        if v is None or str(v).strip()=='': return None
        x=float(str(v).replace(',','.'))
        return x if math.isfinite(x) else None
    except:return None

def read_csv(path,delimiter=','):
    with open(path,'r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f,delimiter=delimiter))
def write_csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:
        fields=[]
        for r in rows:
            for k in r:
                if k not in fields: fields.append(k)
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify_stage0b(tmp):
    required=['STATUS.json','SUCCESS_MARKER.json','OUTPUT_VERIFICATION.json','SCIENTIFIC_GATE_DECISION.json','data/mechanical_master_candidate.csv']
    missing=[x for x in required if not (tmp/x).exists()]
    if missing: raise RuntimeError('Missing Stage0B inputs: '+', '.join(missing))
    status=json.loads((tmp/'STATUS.json').read_text(encoding='utf-8-sig'))
    marker=json.loads((tmp/'SUCCESS_MARKER.json').read_text(encoding='utf-8-sig'))
    ver=json.loads((tmp/'OUTPUT_VERIFICATION.json').read_text(encoding='utf-8-sig'))
    if str(status.get('status','')).lower() not in ('completed','success','passed'): raise RuntimeError('Stage0B status not complete')
    if not any(bool(marker.get(k)) for k in ('success','passed','stage_completed')): raise RuntimeError('Stage0B success marker not true')
    if not any(bool(ver.get(k)) for k in ('verification_passed','passed','success')): raise RuntimeError('Stage0B verification not passed')

def gbif_available():
    try:
        req=urllib.request.Request('https://api.gbif.org/v1/species/match?name=Pinus+nigra',headers={'User-Agent':'WoodPhys-DG/1.0 connectivity check'})
        with urllib.request.urlopen(req,timeout=5) as r:
            return 200 <= getattr(r,'status',200) < 300
    except Exception:
        return False

def gbif_match(name,cache):
    if not name:return None
    if name in cache:return cache[name]
    q=urllib.parse.urlencode({'name':name,'strict':'false'})
    url='https://api.gbif.org/v1/species/match?'+q
    req=urllib.request.Request(url,headers={'User-Agent':'WoodPhys-DG/1.0 research taxonomy audit'})
    try:
        with urllib.request.urlopen(req,timeout=15) as r:d=json.load(r)
        out={k:d.get(k) for k in ('usageKey','scientificName','canonicalName','rank','status','confidence','matchType','kingdom','phylum','class','order','family','genus','species')}
    except Exception as e:
        out={'error':type(e).__name__+': '+str(e)}
    cache[name]=out
    time.sleep(0.05)
    return out

def main():
    if OUT.exists(): shutil.rmtree(OUT)
    for d in ['data','reports','taxonomy','quarantine','input_snapshot']:(OUT/d).mkdir(parents=True,exist_ok=True)
    (OUT/'RUN_LOG.txt').write_text('',encoding='utf-8')
    (OUT/'ERROR_LOG.txt').write_text('',encoding='utf-8')
    write_json(OUT/'STATUS.json',{'stage':'STAGE0C','version':VERSION,'status':'running','started_at':now()})
    shutil.copy2(INPUT_ZIP,OUT/'input_snapshot'/'STAGE0B_VERIFIED_INPUT.zip')
    tmp=OUT/'_stage0b'
    with zipfile.ZipFile(INPUT_ZIP) as z:z.extractall(tmp)
    verify_stage0b(tmp); log('Verified Stage0B input package')
    base_rows=read_csv(tmp/'data'/'mechanical_master_candidate.csv')
    corrected=[]; fig=[]; usda=[]
    for r in base_rows:
        r=dict(r)
        sid=r.get('source_id','')
        if sid.startswith('USDA'):
            r['target_compatibility_group']='CLEAR_WOOD_LONGITUDINAL_STATIC_BENDING'
            r['mechanical_regime']='clear_straight_grained_species_mean'
            r['orientation_standardized']='longitudinal_static_bending_or_table_definition'
            r['pool_directly_with_other_groups']='NO'
            usda.append(r)
        elif sid.startswith('FIGSHARE'):
            r['target_compatibility_group']='CLEAR_WOOD_RADIAL_BENDING'
            r['mechanical_regime']='radial_individual_tree_mean'
            r['orientation_standardized']='radial'
            r['pool_directly_with_other_groups']='NO'
            fig.append(r)
        else:
            r['target_compatibility_group']='UNRESOLVED'
            r['mechanical_regime']='unresolved'
            r['pool_directly_with_other_groups']='NO'
        corrected.append(r)
    write_csv(OUT/'data'/'stage0b_corrected_regime_registry.csv',corrected)
    write_csv(OUT/'data'/'usda_clearwood_longitudinal.csv',usda)
    write_csv(OUT/'data'/'pinus_nigra_radial.csv',fig)
    log(f'Separated Stage0B into USDA longitudinal n={len(usda)} and Pinus radial n={len(fig)}')

    # Spruce: two image faces per board. Retain one physical-measurement row, validate labels agree.
    raw=read_csv(SPRUCE,delimiter=';')
    by={}
    for r in raw:by.setdefault(r['sample_name'],[]).append(r)
    boards=[]; quarantine=[]; pair_issues=[]
    for sample,grp in sorted(by.items()):
        if len(grp)!=2: pair_issues.append({'sample_name':sample,'issue':'EXPECTED_TWO_IMAGE_FACES','observed':len(grp)})
        def vals(k):return {x.get(k,'') for x in grp}
        for k in ('moe','mor','split'):
            if len(vals(k))>1:pair_issues.append({'sample_name':sample,'issue':'PAIR_DISAGREEMENT_'+k,'observed':'|'.join(sorted(vals(k)))})
        physical=next((x for x in grp if str(x.get('length','')).strip()),grp[0])
        moe_gpa=fnum(physical.get('moe')); mor=fnum(physical.get('mor')); dens_raw=fnum(physical.get('density'))
        length=fnum(physical.get('length')); width=fnum(physical.get('width')); height=fnum(physical.get('height')); weight=fnum(physical.get('weight'))
        density_kg_m3=(dens_raw*1_000_000 if dens_raw is not None else None)
        density_calc=(weight/(length*width*height)*1_000_000 if None not in (weight,length,width,height) and length*width*height>0 else None)
        invalid=any(str(x.get('invalid','')).strip() not in ('','0','0.0','nan','None') for x in grp)
        rec={
          'record_id':'SPRUCE_'+sample,'source_id':'ZENODO_16925468_SPRUCE_LAMELLAE','source_doi':'10.5281/zenodo.16925468',
          'license':'CC BY 4.0','measurement_level':'structural_board','species_label_raw':'spruce (species not stated in dataset metadata)',
          'scientific_name':'','genus':'Picea','family':'Pinaceae','taxonomy_status':'GENUS_ONLY_DO_NOT_ASSUME_SPECIES',
          'wood_class':'softwood','country_or_region':'Switzerland / provenance not stated in dataset metadata',
          'moisture_state':'conditioned after kiln drying','moisture_percent':'approximately 12','sample_name':sample,
          'quality_face_1':grp[0].get('quality',''),'quality_face_2':grp[-1].get('quality',''),'provided_split':physical.get('split',''),
          'length_mm':length,'width_mm':width,'height_mm':height,'weight_g':weight,
          'density_raw_g_per_mm3':dens_raw,'density_kg_m3':density_kg_m3,'density_recomputed_kg_m3':density_calc,
          'density_abs_diff_kg_m3':(abs(density_kg_m3-density_calc) if density_kg_m3 is not None and density_calc is not None else None),
          'moe_original_gpa':moe_gpa,'moe_mpa':(moe_gpa*1000 if moe_gpa is not None else None),'mor_mpa':mor,
          'test_direction':'longitudinal structural board bending','test_standard_or_method':'standardized four-point bending; see Achatz et al. 2026',
          'target_compatibility_group':'STRUCTURAL_LAMELLA_LONGITUDINAL_4POINT','mechanical_regime':'full_board_structural_lamella',
          'orientation_standardized':'longitudinal board bending','pool_directly_with_other_groups':'NO','invalid_source_flag':1 if invalid else 0,
          'review_status':'QUARANTINE_SOURCE_INVALID' if invalid else 'SOURCE_VALIDATED_BOARD_LEVEL'
        }
        (quarantine if invalid else boards).append(rec)
    write_csv(OUT/'data'/'spruce_lamellae_board_level_model_ready.csv',boards)
    write_csv(OUT/'quarantine'/'spruce_invalid_boards.csv',quarantine)
    write_csv(OUT/'reports'/'SPRUCE_PAIR_QA.csv',pair_issues,fields=['sample_name','issue','observed'])
    log(f'Spruce deduplication: images={len(raw)}, unique boards={len(by)}, model-ready={len(boards)}, quarantine={len(quarantine)}')

    # Taxonomy: preserve raw labels; use existing scientific names first, optional GBIF for unresolved labels.
    cache={}; tax_rows=[]
    unique={}
    for r in corrected:
        label=(r.get('scientific_name') or r.get('species_label_raw') or '').strip()
        unique[label]=r
    internet_success=0
    gbif_ok=gbif_available()
    log(f'GBIF connectivity available={gbif_ok}')
    for label,r in sorted(unique.items()):
        existing=(r.get('scientific_name') or '').strip()
        query=existing or (r.get('species_label_raw') or '').strip()
        m=(gbif_match(query,cache) if query else None) if gbif_ok else {'error':'GBIF_UNAVAILABLE_SKIPPED'}
        if m and not m.get('error') and (m.get('confidence') or 0)>=80: internet_success+=1
        tax_rows.append({
          'raw_label':r.get('species_label_raw',''),'existing_scientific_name':existing,'query':query,
          'gbif_canonical_name':(m or {}).get('canonicalName',''),'gbif_scientific_name':(m or {}).get('scientificName',''),
          'gbif_genus':(m or {}).get('genus',''),'gbif_family':(m or {}).get('family',''),'gbif_rank':(m or {}).get('rank',''),
          'gbif_status':(m or {}).get('status',''),'gbif_confidence':(m or {}).get('confidence',''),'gbif_match_type':(m or {}).get('matchType',''),
          'resolution_decision':'AUTO_ACCEPT_CANDIDATE_REQUIRES_AUDIT' if m and not m.get('error') and (m.get('confidence') or 0)>=95 else 'MANUAL_REVIEW',
          'error':(m or {}).get('error','')
        })
    write_csv(OUT/'taxonomy'/'GBIF_TAXONOMY_CANDIDATES.csv',tax_rows)
    write_json(OUT/'taxonomy'/'GBIF_CACHE.json',cache)
    log(f'Taxonomy candidates written: {len(tax_rows)}; high-confidence API responses={internet_success}')

    groups=[
      {'group':'CLEAR_WOOD_LONGITUDINAL_STATIC_BENDING','source':'USDA FPL GTR 282 Ch5','rows':len(usda),'measurement_level':'species/source-table mean','MOE_unit':'MPa','MOR_unit':'MPa','direct_pooling':'Only with same orientation, material regime, moisture/protocol definition'},
      {'group':'CLEAR_WOOD_RADIAL_BENDING','source':'Figshare Pinus nigra','rows':len(fig),'measurement_level':'individual tree mean','MOE_unit':'MPa','MOR_unit':'MPa','direct_pooling':'No pooling with USDA longitudinal values'},
      {'group':'STRUCTURAL_LAMELLA_LONGITUDINAL_4POINT','source':'Zenodo Spruce Lamellae','rows':len(boards),'measurement_level':'individual structural board','MOE_unit':'converted GPa to MPa','MOR_unit':'MPa','direct_pooling':'No pooling with clear-wood tables'}]
    write_csv(OUT/'reports'/'TARGET_COMPATIBILITY_MATRIX.csv',groups)
    risks=[
      {'risk_id':'C1','severity':'CRITICAL','risk':'Stage0B treated radial Pinus and longitudinal USDA MOE/MOR as one target','control':'Separated into incompatible target groups; direct pooling forbidden','status':'CONTROLLED'},
      {'risk_id':'C2','severity':'CRITICAL','risk':'Spruce file has two image rows per physical board','control':'Deduplicated by sample_name at board level; pair agreement checked','status':'CONTROLLED'},
      {'risk_id':'C3','severity':'HIGH','risk':'Three invalid physical boards / six image rows in source','control':'Excluded from model-ready file and retained in quarantine','status':'CONTROLLED'},
      {'risk_id':'C4','severity':'HIGH','risk':'USDA species means, Pinus tree means, and Spruce boards are non-exchangeable','control':'measurement_level and regime preserved; hierarchical or separate analyses required','status':'OPEN_FOR_DESIGN'},
      {'risk_id':'C5','severity':'HIGH','risk':'Only one independent source in each compatible mechanical regime','control':'Full cross-study DG remains unauthorized; acquire a matching source','status':'OPEN'},
      {'risk_id':'C6','severity':'MEDIUM','risk':'Spruce botanical species is not stated in dataset metadata','control':'Genus Picea only; no silent Picea abies assumption','status':'CONTROLLED'},
      {'risk_id':'C7','severity':'MEDIUM','risk':'USDA common names remain partly unresolved','control':'GBIF candidates plus manual queue; raw labels preserved','status':'OPEN'}]
    write_csv(OUT/'reports'/'SCIENTIFIC_RISK_REGISTER.csv',risks)

    # Decision: three useful sources, but each in a different target regime.
    decision='YELLOW_REDESIGN'
    rationale=('Three validated mechanical sources are available, including 2,522 model-ready spruce boards, but they represent '
               'different orientations, material regimes, and measurement levels. The earlier 513-row pooled-target interpretation is corrected. '
               'Full cross-study HAP-DG is not authorized until at least two independent sources share a compatible target regime.')
    metrics={'stage0b_rows':len(corrected),'usda_longitudinal_rows':len(usda),'pinus_radial_rows':len(fig),
             'spruce_image_rows':len(raw),'spruce_unique_boards':len(by),'spruce_model_ready_boards':len(boards),
             'spruce_quarantine_boards':len(quarantine),'independent_sources':3,'target_compatibility_groups':3,
             'compatible_groups_with_2plus_sources':0,'taxonomy_labels_audited':len(tax_rows),'gbif_high_confidence_responses':internet_success}
    dec={'decision':decision,'decision_code':'YELLOW_REDESIGN_THREE_SOURCES_ZERO_CROSS_STUDY_COMPATIBLE_GROUPS','rationale':rationale,
         'full_training_authorization':'NO','allowed_next_steps':['manual row/taxonomy audit','acquire one matching longitudinal clear-wood source','small source-specific baseline diagnostics','design separate-regime benchmark'],
         'forbidden_next_steps':['pooling the three sources as one regression target','full HAP-DG claim','cross-study claim within a regime','using supplied Spruce train/test split without independent leakage audit'],
         'metrics':metrics,'generated_at':now()}
    write_json(OUT/'SCIENTIFIC_GATE_DECISION.json',dec)
    write_json(OUT/'reports'/'FEASIBILITY_METRICS.json',metrics)
    report=f'''# WOODPHYS_DG_Q1 — Stage 0C Scientific Gate\n\n## القرار: **{decision}**\n\n{rationale}\n\n## التصحيح العلمي الحاسم\n- USDA وFigshare Pinus nigra لا يقيسان الهدف الفيزيائي نفسه: الأول longitudinal/standard clear-wood tables، والثاني radial MOE/MOR.\n- لذلك تم منع دمجهما، وتصحيح تفسير Stage 0B.\n\n## المصدر الثالث\n- 5,050 صف صورة = وجهان لكل لوح.\n- 2,525 لوحًا فريدًا.\n- {len(quarantine)} ألواح معلّمة invalid حُجزت.\n- {len(boards)} لوحًا model-ready بعد الفصل على مستوى اللوح.\n- MOE حُوّل من GPa إلى MPa استنادًا إلى الورقة، وMOR بقي MPa.\n\n## القرار التنفيذي\nلا يبدأ Full HAP-DG. المرحلة التالية يجب أن تحصل على مصدر مستقل متوافق مع USDA أو تعيد تصميم الورقة كـmulti-regime benchmark بتحليلات منفصلة.\n'''
    (OUT/'GATE_REPORT_AR.md').write_text(report,encoding='utf-8')
    write_json(OUT/'CHECKPOINT.json',{'stage':'STAGE0C','completed':True,'decision':decision,'timestamp':now()})

    # Verification and manifests
    required=['STATUS.json','SCIENTIFIC_GATE_DECISION.json','GATE_REPORT_AR.md',
              'data/stage0b_corrected_regime_registry.csv','data/spruce_lamellae_board_level_model_ready.csv','reports/TARGET_COMPATIBILITY_MATRIX.csv','reports/SCIENTIFIC_RISK_REGISTER.csv']
    missing=[x for x in required if not (OUT/x).exists()]
    checks={'required_files_missing':missing,'spruce_board_count_ok':len(by)==2525,'spruce_model_ready_count_ok':len(boards)==2522,
            'source_count_ok':metrics['independent_sources']==3,'direct_pooling_prevented':all(('No pooling' in g['direct_pooling']) or ('Only with same' in g['direct_pooling']) for g in groups),
            'verification_passed':not missing and len(by)==2525 and len(boards)==2522}
    write_json(OUT/'OUTPUT_VERIFICATION.json',checks)
    if not checks['verification_passed']: raise RuntimeError('Output verification failed: '+json.dumps(checks))
    write_json(OUT/'SUCCESS_MARKER.json',{'success':True,'stage':'STAGE0C','version':VERSION,'scientific_decision':decision,'timestamp':now()})
    write_json(OUT/'STATUS.json',{'stage':'STAGE0C','version':VERSION,'status':'completed','started_at':'see RUN_LOG','completed_at':now(),'scientific_decision':decision})

    files=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name not in ('UPLOAD_THIS_STAGE0C_RESULT.zip','FILE_MANIFEST.csv','HASHES_SHA256.csv'):
            files.append({'relative_path':p.relative_to(OUT).as_posix(),'size_bytes':p.stat().st_size})
    write_csv(OUT/'FILE_MANIFEST.csv',files,['relative_path','size_bytes'])
    hashes=[{'relative_path':r['relative_path'],'sha256':sha(OUT/r['relative_path'])} for r in files]
    write_csv(OUT/'HASHES_SHA256.csv',hashes,['relative_path','sha256'])
    upload=OUT/'UPLOAD_THIS_STAGE0C_RESULT.zip'
    with zipfile.ZipFile(upload,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p!=upload:z.write(p,p.relative_to(OUT))
    shutil.rmtree(tmp,ignore_errors=True)
    log(f'Completed. Scientific decision: {decision}')

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True)
        err=traceback.format_exc()
        (OUT/'ERROR_LOG.txt').write_text(err,encoding='utf-8')
        write_json(OUT/'STATUS.json',{'stage':'STAGE0C','version':VERSION,'status':'failed','failed_at':now()})
        print(err,flush=True)
        raise
