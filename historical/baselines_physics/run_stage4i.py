from __future__ import annotations
from pathlib import Path
import datetime as dt
import hashlib,json,math,os,shutil,sys,traceback,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import GroupShuffleSplit,GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'input'; OUT=ROOT/'stage4i_output'; WORK=OUT/'_work'; TABLES=OUT/'tables'; FIG=OUT/'figures'
ZIPS={
 'stage0f':IN/'STAGE0F_VERIFIED_INPUT.zip','stage2c':IN/'STAGE2C_VERIFIED_INPUT.zip',
 'stage3a':IN/'STAGE3A_FIX2_VERIFIED_INPUT.zip','stage3b':IN/'STAGE3B_VERIFIED_INPUT.zip',
 'stage4c':IN/'STAGE4C_VERIFIED_INPUT.zip','stage4h':IN/'STAGE4H_VERIFIED_INPUT.zip'}
DOMAINS=['DEVELOPMENT_US','CALIBRATION_CANADA','EXTERNAL_OTHER_IMPORTS']
TARGETS=['moe_mpa','mor_mpa']; SEEDS=[13,42,97,131,2026]; ALPHA=.10; BOOT=500; MASTER_SEED=20260715
TEST_MODE=os.environ.get('WOODPHYS_STAGE4I_TEST_MODE')=='1'
TREE_N=30 if TEST_MODE else 80
RUN_SEEDS=SEEDS[:1] if TEST_MODE else SEEDS
LOG=[]; ERR=[]

FEATURE_SETS={
 'FULL':['specific_gravity','moisture_percent','moisture_state','wood_class','genus','family'],
 'PHYSICAL_NO_TAXONOMY':['specific_gravity','moisture_percent','moisture_state','wood_class'],
 'NO_MOISTURE':['specific_gravity','wood_class','genus','family'],
 'DENSITY_ONLY':['specific_gravity'],
 'TAXONOMY_ONLY':['wood_class','genus','family'],
}
MODEL_SPECS={
 'RIDGE_FULL':('RIDGE','FULL'),
 'ELASTIC_NET_FULL':('ELASTIC','FULL'),
 'RANDOM_FOREST_FULL':('RF','FULL'),
 'GRADIENT_BOOSTING_FULL':('GB','FULL'),
 'RIDGE_PHYSICAL_NO_TAXONOMY':('RIDGE','PHYSICAL_NO_TAXONOMY'),
 'RIDGE_NO_MOISTURE':('RIDGE','NO_MOISTURE'),
 'RIDGE_DENSITY_ONLY':('RIDGE','DENSITY_ONLY'),
}
MAIN_MODELS=['RIDGE_FULL','ELASTIC_NET_FULL','RANDOM_FOREST_FULL','GRADIENT_BOOSTING_FULL']
ABLATIONS=['RIDGE_PHYSICAL_NO_TAXONOMY','RIDGE_NO_MOISTURE','RIDGE_DENSITY_ONLY']

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def log(x):
 s=f'[{now()}] {x}';LOG.append(s);print(s,flush=True)
def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_json(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def verify(name,z):
 d=WORK/name;d.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(z) as zz:zz.extractall(d)
 s=json.loads((d/'SUCCESS_MARKER.json').read_text(encoding='utf-8'));v=json.loads((d/'OUTPUT_VERIFICATION.json').read_text(encoding='utf-8'))
 if not s.get('success') or not v.get('verification_passed'):raise RuntimeError(f'{name} not verified')
 hp=d/'HASHES_SHA256.csv'
 if hp.exists():
  bad=[]
  for _,r in pd.read_csv(hp).iterrows():
   p=d/r['relative_path'];a=sha256(p) if p.exists() else 'MISSING'
   if a!=r['sha256']:bad.append(r['relative_path'])
  if bad:raise RuntimeError(f'{name} hash mismatch {bad[:5]}')
 return d

def make_ohe():
 try:return OneHotEncoder(handle_unknown='ignore',sparse_output=False)
 except TypeError:return OneHotEncoder(handle_unknown='ignore',sparse=False)
def preprocessor(features):
 nums=[c for c in features if c in ['specific_gravity','moisture_percent']]
 cats=[c for c in features if c not in nums]
 parts=[]
 if nums:parts.append(('num',Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),nums))
 if cats:parts.append(('cat',Pipeline([('imp',SimpleImputer(strategy='constant',fill_value='__MISSING__')),('ohe',make_ohe())]),cats))
 return ColumnTransformer(parts,sparse_threshold=0.0)
def estimator(kind,seed):
 if kind=='RIDGE':return Ridge(alpha=1.0)
 if kind=='ELASTIC':return ElasticNet(alpha=.05,l1_ratio=.5,max_iter=5000,tol=1e-3,random_state=seed)
 if kind=='RF':return RandomForestRegressor(n_estimators=TREE_N,min_samples_leaf=3,max_features=.8,random_state=seed,n_jobs=1)
 if kind=='ET':return ExtraTreesRegressor(n_estimators=TREE_N,min_samples_leaf=3,max_features=1.0,random_state=seed,n_jobs=1)
 if kind=='GB':return GradientBoostingRegressor(n_estimators=60 if TEST_MODE else 180,learning_rate=.05,max_depth=2,min_samples_leaf=3,random_state=seed,loss='huber')
 raise KeyError(kind)
def model_pipeline(name,seed):
 kind,fs=MODEL_SPECS[name];features=FEATURE_SETS[fs]
 return Pipeline([('pre',preprocessor(features)),('model',estimator(kind,seed))]),features

def split_source(df,seed):
 tr=[];ca=[]
 for _,g in df.groupby('domain_role'):
  groups=g['species_group_key'].fillna(g['species_label_raw']).astype(str)
  sp=GroupShuffleSplit(n_splits=1,test_size=.20,random_state=seed);a,b=next(sp.split(g,groups=groups))
  tr.append(g.iloc[a]);ca.append(g.iloc[b])
 return pd.concat(tr,ignore_index=True),pd.concat(ca,ignore_index=True)
def metric(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);iqr=max(float(np.quantile(y,.75)-np.quantile(y,.25)),1e-9)
 return {'n':len(y),'mae':float(mean_absolute_error(y,p)),'rmse':float(mean_squared_error(y,p)**.5),'r2':float(r2_score(y,p)),'nmae_iqr':float(mean_absolute_error(y,p)/iqr)}
def conformal_q(res):
 r=np.sort(np.asarray(res,float));k=min(len(r)-1,max(0,int(math.ceil((len(r)+1)*(1-ALPHA)))-1));return float(r[k])
def winkler(y,lo,hi,alpha=.1):
 y=np.asarray(y);lo=np.asarray(lo);hi=np.asarray(hi);s=hi-lo;s+=np.where(y<lo,2/alpha*(lo-y),0);s+=np.where(y>hi,2/alpha*(y-hi),0);return s

def development_predictions(usda):
 rows=[]
 for target in TARGETS:
  for held in DOMAINS:
   source=usda[usda.domain_role!=held].copy();test=usda[usda.domain_role==held].copy()
   for name in MODEL_SPECS:
    pipe,features=model_pipeline(name,MASTER_SEED);pipe.fit(source[features],source[target]);pred=pipe.predict(test[features])
    for rid,y,p in zip(test.record_id,test[target],pred):rows.append({'target':target,'held_domain':held,'record_id':rid,'model':name,'y_true':y,'y_pred':p})
 return pd.DataFrame(rows)

def summarize_development(pred,stage2c):
 rows=[]
 for (target,model,held),g in pred.groupby(['target','model','held_domain']):rows.append({'target':target,'model':model,'held_domain':held,**metric(g.y_true,g.y_pred)})
 domain=pd.DataFrame(rows)
 # Add frozen and equal from canonical Stage2C outputs.
 fp=pd.read_csv(stage2c/'data/APPLICABILITY_WEIGHTED_DEVELOPMENT_PREDICTIONS.csv')
 for (target,held),g in fp.groupby(['target','held_domain']):rows.append({'target':target,'model':'FROZEN_APPLICABILITY','held_domain':held,**metric(g.y_true,g.y_pred)})
 pc=pd.read_csv(stage2c/'tables/POINT_CANDIDATES_BY_DOMAIN.csv')
 eq=pc[pc.candidate=='EQUAL_SOURCE_ENSEMBLE']
 for _,r in eq.iterrows():rows.append({'target':r.target,'model':'EQUAL_SOURCE_ENSEMBLE','held_domain':r.held_domain,'n':int(r.n),'mae':r.mae,'rmse':r.rmse,'r2':r.r2,'nmae_iqr':r.normalized_mae_iqr})
 domain=pd.DataFrame(rows)
 summary=domain.groupby(['target','model'],as_index=False).agg(mean_nmae=('nmae_iqr','mean'),worst_nmae=('nmae_iqr','max'),mean_mae=('mae','mean'),mean_r2=('r2','mean'))
 summary['mean_nmae_rank']=summary.groupby('target').mean_nmae.rank(method='min')
 summary['worst_nmae_rank']=summary.groupby('target').worst_nmae.rank(method='min')
 return domain,summary

def external_frame(board):
 raw=pd.read_csv(board,encoding='utf-8-sig')
 mapped=pd.DataFrame({'record_id':raw.BoardID.astype(str),'tree':raw.Tree.astype(str),'spacing':raw.Spacing,'specific_gravity':pd.to_numeric(raw.Density,errors='coerce')/1000.,'moisture_percent':np.nan,'moisture_state':'__MISSING__','wood_class':'softwood','genus':'Pinus','family':'Pinaceae','moe_mpa':pd.to_numeric(raw.MOE,errors='coerce'),'mor_mpa':pd.to_numeric(raw.MOR,errors='coerce')});mapped=mapped.dropna(subset=['moe_mpa','mor_mpa','specific_gravity']).reset_index(drop=True);return raw,mapped
def cluster_boot(df,target,model,seed):
 rng=np.random.default_rng(seed);tree_arr=df.tree.astype(str).to_numpy();trees=np.array(sorted(np.unique(tree_arr)));groups={t:np.where(tree_arr==t)[0] for t in trees};y=df.y_true.to_numpy(float);p=df.y_pred.to_numpy(float);vals={k:[] for k in ['mae','rmse','r2']}
 it=300 if TEST_MODE else BOOT
 for _ in range(it):
  sam=rng.choice(trees,len(trees),replace=True);idx=np.concatenate([groups[t] for t in sam]);m=metric(y[idx],p[idx])
  for k in vals:vals[k].append(m[k])
 obs=metric(y,p);out={'target':target,'model':model,'trees':len(trees),'boards':len(df),**obs}
 for k,a in vals.items():out[k+'_cluster_ci_low']=float(np.nanquantile(a,.025));out[k+'_cluster_ci_high']=float(np.nanquantile(a,.975))
 return out

def calibration_residuals(usda,target,name):
 groups=usda['species_group_key'].fillna(usda['species_label_raw']).astype(str)
 unique=groups.nunique();folds=3 if TEST_MODE else 5
 folds=min(folds,unique)
 gkf=GroupKFold(n_splits=folds);res=[]
 for k,(tr,ca) in enumerate(gkf.split(usda,groups=groups)):
  pipe,features=model_pipeline(name,MASTER_SEED+k);pipe.fit(usda.iloc[tr][features],usda.iloc[tr][target]);res.extend(abs(usda.iloc[ca][target].to_numpy()-pipe.predict(usda.iloc[ca][features])))
 return np.asarray(res,float)
def external_models(usda,ext,canonical):
 predrows=[];metrics=[];intervals=[]
 for target in TARGETS:
  for name in MODEL_SPECS:
   pipe,features=model_pipeline(name,MASTER_SEED);pipe.fit(usda[features],usda[target]);p=pipe.predict(ext[features])
   d=pd.DataFrame({'target':target,'record_id':ext.record_id.astype(str),'tree':ext.tree.astype(str),'model':name,'y_true':ext[target],'y_pred':p})
   predrows.extend(d.to_dict('records'));metrics.append(cluster_boot(d,target,name,MASTER_SEED+len(metrics)))
   if name in MAIN_MODELS:
    residuals=calibration_residuals(usda,target,name);q=conformal_q(residuals);lo=p-q;hi=p+q;y=ext[target].to_numpy(float)
    intervals.append({'target':target,'model':name,'calibration_residual_n':len(residuals),'q90':q,'coverage':float(np.mean((y>=lo)&(y<=hi))),'mean_width':float(np.mean(hi-lo)),'mean_winkler':float(np.mean(winkler(y,lo,hi)))})
  c=canonical[canonical.target==target]
  for label,col in [('ERM_CANONICAL','pred_erm'),('EQUAL_SOURCE_ENSEMBLE','pred_equal_ensemble'),('FROZEN_APPLICABILITY','pred_frozen')]:
   d=pd.DataFrame({'target':target,'record_id':c.record_id.astype(str),'tree':c.tree.astype(str),'model':label,'y_true':c.y_true,'y_pred':c[col]})
   predrows.extend(d.to_dict('records'));metrics.append(cluster_boot(d,target,label,MASTER_SEED+len(metrics)))
 return pd.DataFrame(predrows),pd.DataFrame(metrics),pd.DataFrame(intervals)

def board_physics(raw):
 feature_sets={
  'DENSITY_ONLY':['Density'],
  'DENSITY_POSITION':['Density','Spacing','Pos','CA'],
  'FULL_BOARD_PHYSICS':['Density','Spacing','Pos','CA','MFA','DeltaMFA','DeltaDensity','Knots','SSDensity','RW','KnotA'],
 }
 models={'RIDGE':Ridge(alpha=1.0),'GRADIENT_BOOSTING':GradientBoostingRegressor(n_estimators=60 if TEST_MODE else 180,learning_rate=.05,max_depth=2,min_samples_leaf=3,random_state=MASTER_SEED,loss='huber')}
 rows=[];predrows=[]
 complete=raw.dropna(subset=['MOE','MOR','Tree']).copy();groups=complete.Tree.astype(str);gkf=GroupKFold(n_splits=3 if TEST_MODE else 5)
 for target_col,target in [('MOE','moe_mpa'),('MOR','mor_mpa')]:
  for fs,features in feature_sets.items():
   for mn,est in models.items():
    oof=np.full(len(complete),np.nan)
    for tr,te in gkf.split(complete,groups=groups):
     pipe=Pipeline([('pre',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())])),('model',clone(est))]);pipe.fit(complete.iloc[tr][features],complete.iloc[tr][target_col]);oof[te]=pipe.predict(complete.iloc[te][features])
    m=metric(complete[target_col],oof);rows.append({'target':target,'feature_set':fs,'model':mn,'validation':'5-fold GroupKFold by tree','trees':groups.nunique(),'boards':len(complete),**m})
    for rid,t,y,p in zip(complete.BoardID,groups,complete[target_col],oof):predrows.append({'target':target,'feature_set':fs,'model':mn,'record_id':rid,'tree':t,'y_true':y,'y_pred':p})
 return pd.DataFrame(rows),pd.DataFrame(predrows)

def sensitivity(stage2c):
 temp=pd.read_csv(stage2c/'tables/TEMPERATURE_SELECTION.csv');beta=pd.read_csv(stage2c/'tables/UQ_BETA_SUMMARY.csv');rows=[]
 for target,g in temp.groupby('target'):
  s=g[g.selected=='YES'].iloc[0];tol=s.worst_nmae*1.01;near=g[g.worst_nmae<=tol]
  rows.append({'target':target,'parameter':'temperature','selected':s.temperature,'selected_mean_nmae':s.mean_nmae,'selected_worst_nmae':s.worst_nmae,'near_optimal_1pct_count':len(near),'near_optimal_min':near.temperature.min(),'near_optimal_max':near.temperature.max()})
 for target,g in beta.groupby('target'):
  selected=.75 if target=='moe_mpa' else 1.0;s=g[np.isclose(g.beta,selected)].iloc[0]
  rows.append({'target':target,'parameter':'beta','selected':selected,'selected_mean_nmae':np.nan,'selected_worst_nmae':np.nan,'near_optimal_1pct_count':np.nan,'near_optimal_min':np.nan,'near_optimal_max':np.nan,'mean_coverage':s.mean_coverage,'minimum_domain_seed_coverage':s.minimum_domain_seed_coverage,'width_ratio_to_global':s.width_ratio_to_global})
 return pd.DataFrame(rows),temp,beta

def make_figures(devsum,extmet,boardmet,temp,beta,extpred):
 for target in TARGETS:
  d=devsum[devsum.target==target].sort_values('worst_nmae')
  plt.figure(figsize=(9,5));plt.barh(d.model,d.worst_nmae);plt.xlabel('Worst-domain NMAE (lower is better)');plt.title(f'Development baseline audit: {target}');plt.tight_layout();plt.savefig(FIG/f'DEVELOPMENT_WORST_NMAE_{target}.png',dpi=150);plt.close()
  e=extmet[extmet.target==target].sort_values('mae');x=np.arange(len(e));lo=e.mae-e.mae_cluster_ci_low;hi=e.mae_cluster_ci_high-e.mae
  plt.figure(figsize=(10,5));plt.errorbar(x,e.mae,yerr=[lo,hi],fmt='o',capsize=3);plt.xticks(x,e.model,rotation=45,ha='right');plt.ylabel('External MAE with tree-cluster 95% CI');plt.title(f'Fixed post-hoc baselines: {target}');plt.tight_layout();plt.savefig(FIG/f'EXTERNAL_BASELINE_MAE_{target}.png',dpi=150);plt.close()
  b=boardmet[boardmet.target==target].copy();b['label']=b.feature_set+' | '+b.model;b=b.sort_values('mae')
  plt.figure(figsize=(10,5));plt.barh(b.label,b.mae);plt.xlabel('Grouped-CV MAE');plt.title(f'Within-board physics diagnostic: {target}');plt.tight_layout();plt.savefig(FIG/f'BOARD_PHYSICS_DIAGNOSTIC_{target}.png',dpi=150);plt.close()

def main():
 shutil.rmtree(OUT,ignore_errors=True)
 for d in [OUT,WORK,TABLES,FIG]:d.mkdir(parents=True,exist_ok=True)
 write_json(OUT/'STATUS.json',{'stage':'STAGE4I','status':'running','started_at':now()})
 try:
  roots={n:verify(n,z) for n,z in ZIPS.items()};log('All canonical inputs and hashes verified')
  usda=pd.read_csv(roots['stage0f']/'freeze/usda_clearwood_primary_model_ready.csv');man=pd.read_csv(roots['stage0f']/'splits/USDA_DOMAIN_AND_TAXON_SPLIT_MANIFEST.csv');usda=usda.merge(man[['record_id','domain_role']],on='record_id',validate='one_to_one')
  raw,ext=external_frame(roots['stage3a']/'quarantine/Board_Data.csv');canonical=pd.read_csv(roots['stage3b']/'data/EXTERNAL_FROZEN_PREDICTIONS.csv')
  dp=development_predictions(usda);devdomain,devsum=summarize_development(dp,roots['stage2c']);log('Development baselines and ablations complete')
  ep,em,intervals=external_models(usda,ext,canonical);log('External fixed-baseline audit complete')
  bm,bp=board_physics(raw);log('Board-physics grouped diagnostic complete')
  sens,temp,beta=sensitivity(roots['stage2c'])
  dp.to_csv(TABLES/'DEVELOPMENT_POSTHOC_PREDICTIONS.csv',index=False,encoding='utf-8-sig');devdomain.to_csv(TABLES/'DEVELOPMENT_BASELINE_BY_DOMAIN.csv',index=False,encoding='utf-8-sig');devsum.to_csv(TABLES/'DEVELOPMENT_BASELINE_SUMMARY.csv',index=False,encoding='utf-8-sig');ep.to_csv(TABLES/'EXTERNAL_FIXED_BASELINE_PREDICTIONS.csv',index=False,encoding='utf-8-sig');em.to_csv(TABLES/'EXTERNAL_FIXED_BASELINE_CLUSTER_METRICS.csv',index=False,encoding='utf-8-sig');intervals.to_csv(TABLES/'EXTERNAL_CONSTANT_INTERVAL_AUDIT.csv',index=False,encoding='utf-8-sig');bm.to_csv(TABLES/'STRUCTURAL_BOARD_PHYSICS_GROUPED_CV.csv',index=False,encoding='utf-8-sig');bp.to_csv(TABLES/'STRUCTURAL_BOARD_PHYSICS_OOF_PREDICTIONS.csv',index=False,encoding='utf-8-sig');sens.to_csv(TABLES/'FREEZE_SENSITIVITY_SUMMARY.csv',index=False,encoding='utf-8-sig')
  make_figures(devsum,em,bm,temp,beta,ep)
  for fn in ['TEMPERATURE_CURVE_moe_mpa.png','TEMPERATURE_CURVE_mor_mpa.png','UQ_BETA_CURVE_moe_mpa.png','UQ_BETA_CURVE_mor_mpa.png']:
   shutil.copy2(roots['stage2c']/'figures'/fn,FIG/fn)
  # Scientific summaries; no model is promoted as a replacement on the lockbox.
  best_dev={};best_ext={};physics_gain={}
  for target in TARGETS:
   gd=devsum[devsum.target==target].sort_values(['worst_nmae','mean_nmae']).iloc[0];best_dev[target]={'model':gd.model,'worst_nmae':float(gd.worst_nmae),'mean_nmae':float(gd.mean_nmae)}
   ge=em[em.target==target].sort_values('mae').iloc[0];best_ext[target]={'model':ge.model,'mae':float(ge.mae),'ci':[float(ge.mae_cluster_ci_low),float(ge.mae_cluster_ci_high)]}
   b=bm[bm.target==target];base=b[(b.feature_set=='DENSITY_ONLY')&(b.model=='RIDGE')].iloc[0];full=b[b.feature_set=='FULL_BOARD_PHYSICS'].sort_values('mae').iloc[0];physics_gain[target]={'density_only_ridge_mae':float(base.mae),'best_full_physics_model':full.model,'best_full_physics_mae':float(full.mae),'relative_mae_reduction_percent':float(100*(base.mae-full.mae)/base.mae)}
  decision='POST_HOC_BASELINES_AND_PHYSICS_DIAGNOSTIC_COMPLETE_MANUSCRIPT_REWRITE_REQUIRED'
  write_json(OUT/'SCIENTIFIC_STAGE4I_DECISION.json',{'decision':decision,'analysis_status':'POST_HOC_REVIEWER_MOTIVATED','frozen_method_changed':False,'external_lockbox_used_for_retuning':False,'best_development_descriptive':best_dev,'best_external_descriptive_not_for_selection':best_ext,'board_physics_diagnostic':physics_gain,'same_regime_lockbox_status':'WAITING_FOR_EUCALYPTUS_RESPONSE','interpretation':['Standard model baselines and feature ablations are descriptive post-hoc checks.','No model is selected or retuned using Pinus patula.','Grouped board-physics CV diagnoses whether omitted physical variables can explain transfer failure.','A successful enriched board model does not validate transfer from USDA clear wood.','Eucalyptus remains the required independent same-regime lockbox for the strongest evidence design.'],'next_stage':'STAGE4J_REVISED_MANUSCRIPT_WITH_CLUSTER_BASELINE_AND_PHYSICS_RESULTS','generated_at':now()})
  lines=['# WoodPhys-DG — Stage 4I','',f'## القرار: **{decision}**','','لا توجد Dataset جديدة، ولم تتغير الطريقة المجمدة. كل المقارنات الجديدة post hoc.','','## أفضل نتائج Development الوصفية']
  for t,v in best_dev.items():lines.append(f"- {t}: {v['model']}؛ worst NMAE={v['worst_nmae']:.4f}؛ mean NMAE={v['mean_nmae']:.4f}")
  lines+=['','## أفضل External baseline وصفيًا — لا يستخدم لإعادة الاختيار']
  for t,v in best_ext.items():lines.append(f"- {t}: {v['model']}؛ MAE={v['mae']:.4f}؛ tree-cluster CI [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]")
  lines+=['','## تشخيص متغيرات الألواح الفيزيائية']
  for t,v in physics_gain.items():lines.append(f"- {t}: density-only Ridge MAE={v['density_only_ridge_mae']:.4f}؛ أفضل full-physics={v['best_full_physics_model']} MAE={v['best_full_physics_mae']:.4f}؛ الانخفاض النسبي={v['relative_mae_reduction_percent']:.2f}%")
  lines+=['','إذا تحسن full-board physics بوضوح، فهذا يدعم تفسير أن العقد وMFA وخصائص اللوح المفقودة تسبب جزءًا مهمًا من فشل النقل، لكنه ليس external validation للطريقة المجمدة.','','## التالي','إعادة كتابة البحث في Stage 4J باستخدام نتائج Stage 4H و4I، مع predicted-vs-observed وshift وcluster-aware tables، مع إبقاء Eucalyptus كـsame-regime lockbox منتظر.']
  (OUT/'STAGE4I_REPORT_AR.md').write_text('\n'.join(lines),encoding='utf-8')
  write_json(OUT/'OUTPUT_VERIFICATION.json',{'verification_passed':True,'all_input_hashes_verified':True,'development_fixed_baselines_created':True,'external_cluster_metrics_created':True,'external_constant_interval_audit_created':True,'feature_ablations_created':True,'board_physics_grouped_cv_created':True,'freeze_sensitivity_created':True,'figures_created':len(list(FIG.glob('*.png'))),'frozen_method_changed':False,'external_lockbox_used_for_retuning':False,'test_mode':TEST_MODE})
  write_json(OUT/'STATUS.json',{'stage':'STAGE4I','status':'completed','decision':decision,'completed_at':now()});write_json(OUT/'SUCCESS_MARKER.json',{'success':True,'stage':'STAGE4I','decision':decision,'timestamp':now()});(OUT/'ERROR_LOG.txt').write_text('',encoding='utf-8');log('Stage4I complete');(OUT/'RUN_LOG.txt').write_text('\n'.join(LOG)+'\n',encoding='utf-8')
  manifest=[];hashes=[]
  for p in sorted(OUT.rglob('*')):
   if not p.is_file() or '_work' in p.parts:continue
   rel=p.relative_to(OUT).as_posix()
   if rel in {'FILE_MANIFEST.csv','HASHES_SHA256.csv','UPLOAD_THIS_STAGE4I_RESULT.zip'}:continue
   manifest.append({'relative_path':rel,'size_bytes':p.stat().st_size});hashes.append({'relative_path':rel,'sha256':sha256(p)})
  pd.DataFrame(manifest).to_csv(OUT/'FILE_MANIFEST.csv',index=False,encoding='utf-8-sig');pd.DataFrame(hashes).to_csv(OUT/'HASHES_SHA256.csv',index=False,encoding='utf-8-sig')
  with zipfile.ZipFile(OUT/'UPLOAD_THIS_STAGE4I_RESULT.zip','w',zipfile.ZIP_DEFLATED) as z:
   for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name!='UPLOAD_THIS_STAGE4I_RESULT.zip' and '_work' not in p.parts:z.write(p,p.relative_to(OUT))
  return 0
 except Exception as e:
  ERR.append(traceback.format_exc());write_json(OUT/'STATUS.json',{'stage':'STAGE4I','status':'failed','error':str(e),'completed_at':now()});write_json(OUT/'SUCCESS_MARKER.json',{'success':False,'stage':'STAGE4I','error':str(e),'timestamp':now()});(OUT/'ERROR_LOG.txt').write_text('\n'.join(ERR),encoding='utf-8');(OUT/'RUN_LOG.txt').write_text('\n'.join(LOG)+'\n',encoding='utf-8');print(ERR[-1],file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
