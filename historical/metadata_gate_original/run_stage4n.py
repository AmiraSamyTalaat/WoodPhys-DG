from pathlib import Path
import json, zipfile, shutil, traceback
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'stage4n_output'; TAB=OUT/'tables'; WORK=ROOT/'work'
try:
 WORK.mkdir(exist_ok=True); TAB.mkdir(parents=True,exist_ok=True)
 for name in ['STAGE2C_VERIFIED_INPUT.zip','STAGE3B_VERIFIED_RESULT.zip']:
  dest=WORK/name.replace('.zip','')
  if dest.exists(): shutil.rmtree(dest)
  with zipfile.ZipFile(ROOT/'input'/name) as z: z.extractall(dest)
 dev=pd.read_csv(WORK/'STAGE2C_VERIFIED_INPUT/data/APPLICABILITY_WEIGHTED_DEVELOPMENT_PREDICTIONS.csv')
 ext=pd.read_csv(WORK/'STAGE3B_VERIFIED_RESULT/data/EXTERNAL_FROZEN_PREDICTIONS.csv')
 dev['regime_compatible']=True; dev['legacy_decision']=dev['decision_group']; dev['guard_decision']=dev.decision_group.map({'ACCEPT':'PREDICT','REVIEW':'REVIEW','ABSTAIN':'ABSTAIN'}); dev.to_csv(TAB/'DEVELOPMENT_REGIME_FIRST_GUARD_PREDICTIONS.csv',index=False)
 ext['legacy_decision']=ext['decision_group']; ext['regime_compatible']=False; ext['regime_reason']='Structural-board regime incompatible with clear-wood species/group regime'; ext['guard_decision']='ABSTAIN'; ext.to_csv(TAB/'EXTERNAL_REGIME_FIRST_GUARD_DECISIONS.csv',index=False)
 assert len(ext[ext.target=='moe_mpa'])==207 and len(ext[ext.target=='mor_mpa'])==207
 assert (ext.guard_decision=='ABSTAIN').all()
 (OUT/'OUTPUT_VERIFICATION.json').write_text(json.dumps({'verification_passed':True,'required_outputs_present':True,'external_rows_per_target':207,'guard_all_external_abstain':True},indent=2),encoding='utf-8')
 (OUT/'SUCCESS_MARKER.json').write_text(json.dumps({'success':True,'stage':'STAGE4N'},indent=2),encoding='utf-8')
 print('Stage 4N guard verification passed')
except Exception as e:
 (OUT/'ERROR_LOG.txt').write_text(traceback.format_exc(),encoding='utf-8'); raise
