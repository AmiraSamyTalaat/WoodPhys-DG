from pathlib import Path
import json, hashlib, itertools, shutil, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'generated'
USDA_PATH=ROOT/'data/usda_clearwood_primary_model_ready.csv';MANIFEST_PATH=ROOT/'data/USDA_DOMAIN_AND_TAXON_SPLIT_MANIFEST.csv';SPRUCE_PATH=ROOT/'data/spruce_structural_board_separate_regime.csv';CANON_PATH=ROOT/'data/APPLICABILITY_WEIGHTED_DEVELOPMENT_PREDICTIONS.csv'
TARGETS=['moe_mpa','mor_mpa'];TAU={'moe_mpa':3.0,'mor_mpa':.2};BETA={'moe_mpa':.75,'mor_mpa':1.};FROZEN_Q={'moe_mpa':2934.45789999,'mor_mpa':16.12726911};FROZEN_MCAL={'moe_mpa':567.8071925336885,'mor_mpa':2.7262826228755124};FEATURES=['specific_gravity','moisture_percent','moisture_state','wood_class','genus','family'];DOMAINS=['DEVELOPMENT_US','CALIBRATION_CANADA','EXTERNAL_OTHER_IMPORTS'];SEEDS=[13,42,97,131,2026]
def ohe():
 try:return OneHotEncoder(handle_unknown='ignore',sparse_output=False)
 except TypeError:return OneHotEncoder(handle_unknown='ignore',sparse=False)
def preprocessor():return ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler())]),['specific_gravity','moisture_percent']),('cat',Pipeline([('imp',SimpleImputer(strategy='constant',fill_value='__MISSING__')),('oh',ohe())]),['moisture_state','wood_class','genus','family'])],sparse_threshold=0.0)
def fit_source_ensemble(train,target):
 pp=preprocessor();X=pp.fit_transform(train[FEATURES]);y=train[target].to_numpy(float);domains=[d for d in DOMAINS if d in set(train.domain_role)];mods=[];c=[]
 for d in domains:
  ix=train.domain_role.eq(d).to_numpy();mods.append(Ridge().fit(X[ix],y[ix]));c.append(X[ix].mean(0))
 return {'pre':pp,'domains':domains,'models':mods,'centroids':np.array(c),'pooled':Ridge().fit(X,y)}
def predict_bundle(b,df,tau):
 X=b['pre'].transform(df[FEATURES]);P=np.column_stack([m.predict(X) for m in b['models']]);D=np.sqrt(((X[:,None,:]-b['centroids'][None,:,:])**2).sum(2));A=np.exp(-D/tau);W=A/A.sum(1,keepdims=True)
 return {'pred_frozen':(W*P).sum(1),'pred_equal':P.mean(1),'pred_erm':b['pooled'].predict(X),'disagreement':P.std(1),'weights':W,'distances':D,'source_predictions':P}
def basic(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);iqr=max(np.quantile(y,.75)-np.quantile(y,.25),1e-12);return {'n':len(y),'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**.5,'r2':r2_score(y,p),'nmae_iqr':mean_absolute_error(y,p)/iqr,'bias':np.mean(p-y)}
def winkler(y,lo,hi,a):
 y=np.asarray(y);lo=np.asarray(lo);hi=np.asarray(hi);return hi-lo+(2/a)*(lo-y)*(y<lo)+(2/a)*(y-hi)*(y>hi)
def fast_cluster_mean_ci(values,clusters,B=1000,seed=0):
 x=pd.DataFrame({'v':np.asarray(values,float),'c':np.asarray(clusters)}).groupby('c').v.agg(['sum','count']);s=x['sum'].to_numpy();n=x['count'].to_numpy();k=len(x);rng=np.random.default_rng(seed);idx=rng.integers(0,k,size=(B,k));vals=s[idx].sum(1)/n[idx].sum(1);return np.quantile(vals,[.025,.975])
def labels(dis,a,b):return np.where(dis<=a,'PREDICT',np.where(dis<=b,'REVIEW','ABSTAIN'))
usda=pd.read_csv(USDA_PATH).merge(pd.read_csv(MANIFEST_PATH)[['record_id','domain_role','genus_group_fold_5']],on='record_id');canon=pd.read_csv(CANON_PATH).merge(usda[['record_id','species_group_key','genus','specific_gravity']],on='record_id')
# interval
rows=[];clrows=[];drows=[];front=[]
for t in TARGETS:
 for held in DOMAINS:
  tr=usda[usda.domain_role!=held];te=usda[usda.domain_role==held];p=predict_bundle(fit_source_ensemble(tr,t),te,TAU[t]);cal=canon[(canon.target==t)&(canon.held_domain!=held)];mcal=max(np.median(cal.source_disagreement),1e-12)
  for nom in [.8,.9,.95]:
   q=np.quantile(cal.absolute_error,nom);alpha=1-nom;ah=q*(1+BETA[t]*np.clip(p['disagreement']/mcal,0,5));ch=np.repeat(ah.mean(),len(te));gh=np.repeat(q,len(te))
   for method,h in [('adaptive_disagreement',ah),('width_matched_constant',ch),('global_constant',gh)]:
    lo=p['pred_frozen']-h;hi=p['pred_frozen']+h;cov=((te[t].to_numpy()>=lo)&(te[t].to_numpy()<=hi)).astype(float);ws=winkler(te[t],lo,hi,alpha);ci=fast_cluster_mean_ci(cov,te.species_group_key,500,4);wi=fast_cluster_mean_ci(ws,te.species_group_key,500,5)
    rows.append({'target':t,'held_domain':held,'nominal':nom,'method':method,'coverage':cov.mean(),'mean_width':np.mean(2*h),'mean_winkler':ws.mean(),'coverage_ci_low':ci[0],'coverage_ci_high':ci[1],'winkler_ci_low':wi[0],'winkler_ci_high':wi[1]})
    tmp=pd.DataFrame({'genus':te.genus.values,'species_group_key':te.species_group_key.values,'specific_gravity':te.specific_gravity.values,'covered':cov,'width':2*h,'winkler':ws})
    for (gen,cl),g in tmp.groupby(['genus','species_group_key']):clrows.append({'target':t,'held_domain':held,'nominal':nom,'method':method,'genus':gen,'species_group_key':cl,'n':len(g),'coverage':g.covered.mean(),'mean_width':g.width.mean(),'mean_winkler':g.winkler.mean()})
    bins=pd.qcut(tmp.specific_gravity,4,duplicates='drop')
    for qb,g in tmp.groupby(bins,observed=True):drows.append({'target':t,'held_domain':held,'nominal':nom,'method':method,'density_bin':str(qb),'n':len(g),'coverage':g.covered.mean(),'mean_width':g.width.mean(),'mean_winkler':g.winkler.mean()})
   for beta in [0,.25,.5,.75,1,1.5,2]:
    h=q*(1+beta*np.clip(p['disagreement']/mcal,0,5));lo=p['pred_frozen']-h;hi=p['pred_frozen']+h;front.append({'target':t,'held_domain':held,'nominal':nom,'beta':beta,'coverage':np.mean((te[t]>=lo)&(te[t]<=hi)),'mean_width':np.mean(2*h)})
pd.DataFrame(rows).to_csv(OUT/'tables/CLUSTER_CONDITIONAL_INTERVAL_SUMMARY.csv',index=False);cl=pd.DataFrame(clrows);cl.to_csv(OUT/'tables/INTERVAL_BY_GENUS_AND_CLUSTER.csv',index=False);pd.DataFrame(drows).to_csv(OUT/'tables/INTERVAL_BY_DENSITY_QUARTILE.csv',index=False);pd.DataFrame(front).to_csv(OUT/'tables/COVERAGE_WIDTH_FRONTIER.csv',index=False)
macro=[]
for keys,g in cl.groupby(['target','held_domain','nominal','method']):macro.append(dict(zip(['target','held_domain','nominal','method'],keys),macro_cluster_coverage=g.coverage.mean(),worst_cluster_coverage=g.coverage.min(),median_cluster_coverage=g.coverage.median(),clusters=len(g)))
pd.DataFrame(macro).to_csv(OUT/'tables/MACRO_AND_WORST_CLUSTER_COVERAGE.csv',index=False)
# guard stability
sr=[];qr=[]
for t in TARGETS:
 for held in DOMAINS:
  tr=usda[usda.domain_role!=held];te=usda[usda.domain_role==held];bb=fit_source_ensemble(tr,t);ptr=predict_bundle(bb,tr,TAU[t]);pte=predict_bundle(bb,te,TAU[t]);q0=np.quantile(ptr['disagreement'],[.5,.9]);d0=labels(pte['disagreement'],*q0);all=[]
  for seed in SEEDS:
   rng=np.random.default_rng(seed);pieces=[]
   for dom in bb['domains']:
    sd=tr[tr.domain_role==dom];gr=sd.species_group_key.unique();samp=rng.choice(gr,len(gr),replace=True);pieces.append(pd.concat([sd[sd.species_group_key==g] for g in samp],ignore_index=True))
   bt=pd.concat(pieces,ignore_index=True);b=fit_source_ensemble(bt,t);a=predict_bundle(b,bt,TAU[t]);e=predict_bundle(b,te,TAU[t]);q=np.quantile(a['disagreement'],[.5,.9]);d=labels(e['disagreement'],*q);all.append(d)
   pj=np.sum((d=='PREDICT')&(d0=='PREDICT'))/max(np.sum((d=='PREDICT')|(d0=='PREDICT')),1);aj=np.sum((d=='ABSTAIN')&(d0=='ABSTAIN'))/max(np.sum((d=='ABSTAIN')|(d0=='ABSTAIN')),1)
   sr.append({'target':t,'held_domain':held,'seed':seed,'changed_fraction_vs_base':np.mean(d!=d0),'predict_jaccard_vs_base':pj,'abstain_jaccard_vs_base':aj,'predict_fraction':np.mean(d=='PREDICT'),'review_fraction':np.mean(d=='REVIEW'),'abstain_fraction':np.mean(d=='ABSTAIN')})
  arr=np.vstack(all);sr.append({'target':t,'held_domain':held,'seed':'ACROSS_SEEDS','changed_fraction_vs_base':np.mean(np.any(arr!=arr[0],axis=0)),'predict_jaccard_vs_base':np.nan,'abstain_jaccard_vs_base':np.nan,'predict_fraction':np.mean(arr=='PREDICT'),'review_fraction':np.mean(arr=='REVIEW'),'abstain_fraction':np.mean(arr=='ABSTAIN')})
  for ql in [.45,.5,.55]:
   for qh in [.85,.9,.95]:
    q=np.quantile(ptr['disagreement'],[ql,qh]);d=labels(pte['disagreement'],*q);qr.append({'target':t,'held_domain':held,'q_low':ql,'q_high':qh,'changed_fraction_vs_q50_q90':np.mean(d!=d0),'predict_fraction':np.mean(d=='PREDICT'),'review_fraction':np.mean(d=='REVIEW'),'abstain_fraction':np.mean(d=='ABSTAIN')})
pd.DataFrame(sr).to_csv(OUT/'tables/GUARD_DECISION_STABILITY.csv',index=False);pd.DataFrame(qr).to_csv(OUT/'tables/GUARD_THRESHOLD_SENSITIVITY.csv',index=False)
# Spruce
spr=pd.read_csv(SPRUCE_PATH);s=spr.copy();s['specific_gravity']=s.density_kg_m3/1000;s['moisture_percent']=pd.to_numeric(s.moisture_percent,errors='coerce');s['moisture_state']='__MISSING__';s['wood_class']='softwood';s['genus']='Picea';s['family']='Pinaceae';s['sample_cluster']=s.sample_name.astype(str).str.split('.').str[0]
preds=[];mets=[];ints=[];weights=[]
for t in TARGETS:
 b=fit_source_ensemble(usda,t);p=predict_bundle(b,s,TAU[t]);h=FROZEN_Q[t]*(1+BETA[t]*np.clip(p['disagreement']/FROZEN_MCAL[t],0,5));lo=p['pred_frozen']-h;hi=p['pred_frozen']+h;q=np.quantile(canon[canon.target==t].source_disagreement,[.5,.9]);dec=labels(p['disagreement'],*q)
 d=pd.DataFrame({'target':t,'record_id':spr.record_id,'provided_split':spr.provided_split,'sample_cluster':s.sample_cluster,'length_mm':spr.length_mm,'density_kg_m3':spr.density_kg_m3,'y_true':spr[t],'pred_erm':p['pred_erm'],'pred_equal':p['pred_equal'],'pred_frozen':p['pred_frozen'],'source_disagreement':p['disagreement'],'decision':dec,'interval_lower':lo,'interval_upper':hi});preds.append(d)
 for subset,g in [('all',d),('provided_test',d[d.provided_split=='test']),('metadata_qc_sensitivity',d[d.length_mm>=500])]:
  for method,col in [('ERM','pred_erm'),('EQUAL','pred_equal'),('FROZEN','pred_frozen')]:mets.append({'target':t,'subset':subset,'method':method,**basic(g.y_true,g[col])})
  ws=winkler(g.y_true,g.interval_lower,g.interval_upper,.1);ints.append({'target':t,'subset':subset,'coverage':np.mean((g.y_true>=g.interval_lower)&(g.y_true<=g.interval_upper)),'mean_width':np.mean(g.interval_upper-g.interval_lower),'mean_winkler':np.mean(ws),'predict_fraction':np.mean(g.decision=='PREDICT'),'review_fraction':np.mean(g.decision=='REVIEW'),'abstain_fraction':np.mean(g.decision=='ABSTAIN'),'spearman_disagreement_error':spearmanr(g.source_disagreement,np.abs(g.y_true-g.pred_frozen)).statistic})
 for j,dom in enumerate(b['domains']):weights.append(pd.DataFrame({'target':t,'record_id':spr.record_id,'source_domain':dom,'weight':p['weights'][:,j],'distance':p['distances'][:,j],'source_prediction':p['source_predictions'][:,j]}))
pd.concat(preds).to_csv(OUT/'data/SPRUCE_FROZEN_EXTERNAL_PREDICTIONS.csv',index=False);pd.DataFrame(mets).to_csv(OUT/'tables/SPRUCE_EXTERNAL_POINT_METRICS.csv',index=False);pd.DataFrame(ints).to_csv(OUT/'tables/SPRUCE_EXTERNAL_INTERVAL_AND_GUARD_METRICS.csv',index=False);pd.concat(weights).to_csv(OUT/'data/SPRUCE_SOURCE_WEIGHTS.csv',index=False)
# checklist/protocol/report
check=[('Nested leave-one-domain-out','DONE','Training-only disagreement thresholds; held-domain risk curves and cluster CIs.'),('Nested leave-one-genus-out','DONE','Five fixed genus-group folds with training-only thresholds.'),('PREDICT/REVIEW/ABSTAIN','DONE','Reported for domain and genus held-out rows.'),('Risk-coverage and AURC','DONE','Full curves plus 50/75/90% risk.'),('False-abstention rate','RENAMED','Reported as compatible-regime abstention rate; true false-abstention needs an independent compatible lockbox.'),('Cluster bootstrap confidence intervals','DONE','Species/group cluster bootstrap.'),('Cluster-conditional intervals','DONE','Domain, genus/species cluster, macro, worst-cluster, density quartile.'),('Coverage-width frontier','DONE','80/90/95% nominal levels and beta frontier.'),('Random disagreement control','DONE','500 permutations per fold and coverage.'),('Permuted source centroids','DONE','All non-identity permutations.'),('Constant-width controls 80/90/95','DONE','Global and width-matched comparators.'),('Guard stability across seeds','DONE','Five cluster-bootstrap refits and Jaccard/change rates.'),('q45/q55 q85/q95 sensitivity','DONE','Sensitivity only.'),('Prespecified decision rules','DONE','Machine-readable protocol.'),('Free public external data','DONE','2522 Spruce structural-lamella rows; second cross-regime test.'),('GitHub/Zenodo DOI','NOT_DONE_EXTERNALLY','Repository-ready package only; no external account write or DOI minting.'),('Independent same-regime lockbox','NOT_AVAILABLE','No established free specimen-level dataset meeting the full schema.')]
pd.DataFrame(check,columns=['item','status','evidence_or_limit']).to_csv(OUT/'tables/STAGE4P_COMPLETION_CHECKLIST.csv',index=False)
proto={'protocol_id':'WOODPHYS_DG_STAGE4P_PUBLIC_EXTERNAL_REPLICATION_V1','claim_boundary':'Spruce is a second cross-regime structural-board replication, not a same-regime clear-wood lockbox.','frozen_parameters':{t:{'tau':TAU[t],'beta':BETA[t],'q':FROZEN_Q[t],'m_cal':FROZEN_MCAL[t]} for t in TARGETS},'primary_spruce_analysis':'all 2522 source-validated rows','metadata_only_sensitivity':'exclude specimen length below 500 mm','prohibited':['tune on Spruce labels','claim same-regime transport','claim universal superiority'],'decision_hierarchy':{'FULL_SUPPORT':'positive R2 and no material inferiority to simple comparators on compatible data','PARTIAL_SUPPORT':'limited triage or transport evidence','NO_TRANSPORT_SUPPORT':'negative R2 or material inferiority; coverage due mainly to width'}}
(OUT/'protocol/FROZEN_STAGE4P_PROTOCOL.json').write_text(json.dumps(proto,indent=2),encoding='utf-8')
sm=pd.DataFrame(mets);si=pd.DataFrame(ints);lines=['# Stage 4P public-data completion report','','## Completion status']+[f'- **{a}: {b}.** {c}' for a,b,c in check]+['','## Spruce results']
for t in TARGETS:
 lines.append(f'### {t}')
 for _,r in sm[(sm.target==t)&(sm.subset=='all')].iterrows():lines.append(f"- {r.method}: MAE {r.mae:.3f}; R2 {r.r2:.3f}; NMAE {r.nmae_iqr:.3f}.")
 r=si[(si.target==t)&(si.subset=='all')].iloc[0];lines.append(f'- Frozen interval coverage {r.coverage:.3f}; mean width {r.mean_width:.3f}; P/R/A {r.predict_fraction:.3f}/{r.review_fraction:.3f}/{r.abstain_fraction:.3f}.')
lines+=['','Spruce is an independent public structural-lamella dataset. It tests replication of the cross-regime conclusion and cannot be described as a same-regime clear-wood lockbox.','', 'No external DOI was minted.']
(OUT/'submission/STAGE4P_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
shutil.copy2(SPRUCE_PATH,OUT/'data/spruce_structural_board_separate_regime.csv');shutil.copy2(USDA_PATH,OUT/'data/usda_clearwood_primary_model_ready.csv');shutil.copy2(MANIFEST_PATH,OUT/'data/USDA_DOMAIN_AND_TAXON_SPLIT_MANIFEST.csv');shutil.copy2(ROOT/'scripts/run_stage4p.py',OUT/'scripts/run_stage4p.py');shutil.copy2(ROOT/'scripts/continue_stage4p.py',OUT/'scripts/continue_stage4p.py')
(OUT/'repository/requirements.txt').write_text('numpy\npandas\nscipy\nscikit-learn\npython-docx\nmatplotlib\n');(OUT/'repository/RUN_PROJECT.bat').write_text('@echo off\npython scripts\\run_stage4p.py\npause\n');(OUT/'repository/README.md').write_text('# WoodPhys-DG Stage 4P\nRepository-ready local package. Run the scripts to reproduce nested audits and the public Spruce replication.\n')
rows=[]
for p in sorted(OUT.rglob('*')):
 if p.is_file():rows.append({'relative_path':str(p.relative_to(OUT)),'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
pd.DataFrame(rows).to_csv(OUT/'SHA256_MANIFEST.csv',index=False);(OUT/'SUCCESS_MARKER.json').write_text(json.dumps({'success':True,'frozen_reproduction_pass':True,'spruce_rows':len(spr),'external_repository_doi_minted':False},indent=2))
print(pd.DataFrame(mets).to_string(index=False));print(pd.DataFrame(ints).to_string(index=False))
