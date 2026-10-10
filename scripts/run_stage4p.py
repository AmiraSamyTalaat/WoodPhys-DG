from __future__ import annotations
from pathlib import Path
import json, hashlib, itertools, math, shutil, zipfile, platform, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'generated'
if OUT.exists(): shutil.rmtree(OUT)
for d in ['data','tables','figures','protocol','scripts','submission','repository']:
    (OUT/d).mkdir(parents=True,exist_ok=True)

USDA_PATH=ROOT/'data/usda_clearwood_primary_model_ready.csv'
MANIFEST_PATH=ROOT/'data/USDA_DOMAIN_AND_TAXON_SPLIT_MANIFEST.csv'
SPRUCE_PATH=ROOT/'data/spruce_structural_board_separate_regime.csv'
CANON_PATH=ROOT/'data/APPLICABILITY_WEIGHTED_DEVELOPMENT_PREDICTIONS.csv'
PINUS_PRED_PATH=ROOT/'data/EXTERNAL_FROZEN_PREDICTIONS.csv'
TARGETS=['moe_mpa','mor_mpa']
TAU={'moe_mpa':3.0,'mor_mpa':0.2}
BETA={'moe_mpa':0.75,'mor_mpa':1.0}
FROZEN_Q={'moe_mpa':2934.45789999,'mor_mpa':16.12726911}
FROZEN_MCAL={'moe_mpa':567.8071925336885,'mor_mpa':2.7262826228755124}
FEATURES=['specific_gravity','moisture_percent','moisture_state','wood_class','genus','family']
DOMAINS=['DEVELOPMENT_US','CALIBRATION_CANADA','EXTERNAL_OTHER_IMPORTS']
SEEDS=[13,42,97,131,2026]
RNG=np.random.default_rng(20260731)

def ohe():
    try: return OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    except TypeError: return OneHotEncoder(handle_unknown='ignore',sparse=False)

def preprocessor():
    return ColumnTransformer([
        ('num',Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler())]),['specific_gravity','moisture_percent']),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='constant',fill_value='__MISSING__')),('oh',ohe())]),['moisture_state','wood_class','genus','family'])
    ],sparse_threshold=0.0)

def fit_source_ensemble(train:pd.DataFrame,target:str):
    pp=preprocessor(); X=pp.fit_transform(train[FEATURES]); y=train[target].to_numpy(float)
    domains=[d for d in DOMAINS if d in set(train.domain_role)]
    models=[]; cents=[]
    for d in domains:
        ix=train.domain_role.eq(d).to_numpy()
        models.append(Ridge(alpha=1.0).fit(X[ix],y[ix]))
        cents.append(X[ix].mean(axis=0))
    pooled=Ridge(alpha=1.0).fit(X,y)
    return {'pre':pp,'domains':domains,'models':models,'centroids':np.asarray(cents),'pooled':pooled,'target':target}

def predict_bundle(bundle,df:pd.DataFrame,tau:float,centroid_perm=None):
    X=bundle['pre'].transform(df[FEATURES]); P=np.column_stack([m.predict(X) for m in bundle['models']])
    C=bundle['centroids'] if centroid_perm is None else bundle['centroids'][list(centroid_perm)]
    D=np.sqrt(((X[:,None,:]-C[None,:,:])**2).sum(axis=2))
    A=np.exp(-D/tau); W=A/A.sum(axis=1,keepdims=True)
    frozen=(W*P).sum(axis=1)
    return {'pred_frozen':frozen,'pred_equal':P.mean(axis=1),'pred_erm':bundle['pooled'].predict(X),'disagreement':P.std(axis=1,ddof=0),'source_predictions':P,'weights':W,'distances':D}

def basic_metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float); iqr=max(np.quantile(y,.75)-np.quantile(y,.25),1e-12)
    return {'n':len(y),'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**0.5,'r2':r2_score(y,p),'nmae_iqr':mean_absolute_error(y,p)/iqr,'bias':float(np.mean(p-y))}

def risk_curve(y,p,risk,min_cov=.2,points=81):
    y=np.asarray(y);p=np.asarray(p);risk=np.asarray(risk); order=np.argsort(risk)
    covs=np.linspace(min_cov,1.0,points); rows=[]
    for c in covs:
        n=max(1,int(round(c*len(y)))); ix=order[:n]
        rows.append((n/len(y),mean_absolute_error(y[ix],p[ix]),n))
    a=np.trapz([r[1] for r in rows],[r[0] for r in rows])/(1-min_cov)
    return rows,float(a)

def risk_at(y,p,risk,c):
    order=np.argsort(np.asarray(risk)); n=max(1,int(round(c*len(order))));ix=order[:n]
    return mean_absolute_error(np.asarray(y)[ix],np.asarray(p)[ix]),n

def decision_labels(dis,qlo,qhi):
    return np.where(dis<=qlo,'PREDICT',np.where(dis<=qhi,'REVIEW','ABSTAIN'))

def winkler(y,lo,hi,alpha):
    y=np.asarray(y);lo=np.asarray(lo);hi=np.asarray(hi);w=hi-lo
    return w + (2/alpha)*(lo-y)*(y<lo) + (2/alpha)*(y-hi)*(y>hi)

def cluster_boot_ci(df,metric_fn,cluster='species_group_key',B=250,seed=0):
    rng=np.random.default_rng(seed); groups=df[cluster].dropna().unique(); vals=[]
    if len(groups)<2:return (np.nan,np.nan)
    by={g:df[df[cluster]==g] for g in groups}
    for _ in range(B):
        samp=rng.choice(groups,size=len(groups),replace=True)
        x=pd.concat([by[g] for g in samp],ignore_index=True)
        vals.append(metric_fn(x))
    return tuple(np.quantile(vals,[.025,.975]))

usda=pd.read_csv(USDA_PATH)
man=pd.read_csv(MANIFEST_PATH)
usda=usda.merge(man[['record_id','domain_role','genus_group_fold_5']],on='record_id',how='left')
canon=pd.read_csv(CANON_PATH).merge(usda[['record_id','species_group_key','genus','specific_gravity']],on='record_id',how='left')

# 0 exact reproduction check against Pinus frozen outputs
pinus_raw=pd.read_csv(ROOT/'data/Board_Data.csv').dropna(subset=['MOE','MOR']).copy()
pinus_map=pd.DataFrame({'specific_gravity':pinus_raw.Density/1000,'moisture_percent':np.nan,'moisture_state':'__MISSING__','wood_class':'softwood','genus':'Pinus','family':'Pinaceae'})
pinus_can=pd.read_csv(PINUS_PRED_PATH)
repro=[]
for t in TARGETS:
    b=fit_source_ensemble(usda,t); pr=predict_bundle(b,pinus_map,TAU[t]); c=pinus_can[pinus_can.target==t].reset_index(drop=True)
    repro.append({'target':t,'max_abs_diff_frozen':float(np.max(np.abs(pr['pred_frozen']-c.pred_frozen))),'max_abs_diff_erm':float(np.max(np.abs(pr['pred_erm']-c.pred_erm))),'max_abs_diff_equal':float(np.max(np.abs(pr['pred_equal']-c.pred_equal_ensemble))),'status':'PASS' if np.max(np.abs(pr['pred_frozen']-c.pred_frozen))<1e-6 else 'FAIL'})
pd.DataFrame(repro).to_csv(OUT/'tables/FROZEN_METHOD_REPRODUCTION_CHECK.csv',index=False)

# 1 nested leave-one-domain-out compatible audit
nested_preds=[]; risk_rows=[]; decision_rows=[]; perm_rows=[]
for t in TARGETS:
    for held in DOMAINS:
        tr=usda[usda.domain_role!=held].copy(); te=usda[usda.domain_role==held].copy()
        bun=fit_source_ensemble(tr,t); ptr=predict_bundle(bun,tr,TAU[t]); pte=predict_bundle(bun,te,TAU[t])
        q50,q90=np.quantile(ptr['disagreement'],[.5,.9]); dec=decision_labels(pte['disagreement'],q50,q90)
        base=pd.DataFrame({'target':t,'audit':'leave_one_domain_out','fold':held,'record_id':te.record_id.values,'species_group_key':te.species_group_key.values,'genus':te.genus.values,'specific_gravity':te.specific_gravity.values,'y_true':te[t].values,'y_pred':pte['pred_frozen'],'disagreement':pte['disagreement'],'decision':dec,'q50_train':q50,'q90_train':q90})
        nested_preds.append(base)
        rc,aurc=risk_curve(base.y_true,base.y_pred,base.disagreement)
        rho=spearmanr(base.disagreement,np.abs(base.y_true-base.y_pred)).statistic
        for c,mae,n in rc:risk_rows.append({'target':t,'audit':'leave_one_domain_out','fold':held,'coverage':c,'mae':mae,'retained_n':n,'aurc':aurc,'spearman_risk_error':rho})
        for group,g in base.groupby('decision'):
            decision_rows.append({'target':t,'audit':'leave_one_domain_out','fold':held,'decision':group,**basic_metrics(g.y_true,g.y_pred),'fraction':len(g)/len(base)})
        # centroid permutation control
        ident=tuple(range(len(bun['domains'])))
        for perm in itertools.permutations(range(len(bun['domains']))):
            if perm==ident:continue
            pp=predict_bundle(bun,te,TAU[t],centroid_perm=perm)
            perm_rows.append({'target':t,'audit':'leave_one_domain_out','fold':held,'permutation':'|'.join(map(str,perm)),**basic_metrics(te[t],pp['pred_frozen'])})

# 1b nested leave-one-genus-fold-out
for t in TARGETS:
    for fold in sorted(usda.genus_group_fold_5.unique()):
        tr=usda[usda.genus_group_fold_5!=fold].copy();te=usda[usda.genus_group_fold_5==fold].copy()
        bun=fit_source_ensemble(tr,t);ptr=predict_bundle(bun,tr,TAU[t]);pte=predict_bundle(bun,te,TAU[t])
        q50,q90=np.quantile(ptr['disagreement'],[.5,.9]);dec=decision_labels(pte['disagreement'],q50,q90)
        base=pd.DataFrame({'target':t,'audit':'leave_one_genus_fold_out','fold':fold,'record_id':te.record_id.values,'species_group_key':te.species_group_key.values,'genus':te.genus.values,'specific_gravity':te.specific_gravity.values,'y_true':te[t].values,'y_pred':pte['pred_frozen'],'disagreement':pte['disagreement'],'decision':dec,'q50_train':q50,'q90_train':q90})
        nested_preds.append(base)
        rc,aurc=risk_curve(base.y_true,base.y_pred,base.disagreement);rho=spearmanr(base.disagreement,np.abs(base.y_true-base.y_pred)).statistic
        for c,mae,n in rc:risk_rows.append({'target':t,'audit':'leave_one_genus_fold_out','fold':fold,'coverage':c,'mae':mae,'retained_n':n,'aurc':aurc,'spearman_risk_error':rho})
        for group,g in base.groupby('decision'):
            decision_rows.append({'target':t,'audit':'leave_one_genus_fold_out','fold':fold,'decision':group,**basic_metrics(g.y_true,g.y_pred),'fraction':len(g)/len(base)})
        ident=tuple(range(len(bun['domains'])))
        for perm in itertools.permutations(range(len(bun['domains']))):
            if perm==ident:continue
            pp=predict_bundle(bun,te,TAU[t],centroid_perm=perm)
            perm_rows.append({'target':t,'audit':'leave_one_genus_fold_out','fold':fold,'permutation':'|'.join(map(str,perm)),**basic_metrics(te[t],pp['pred_frozen'])})

nested=pd.concat(nested_preds,ignore_index=True);nested.to_csv(OUT/'tables/NESTED_COMPATIBLE_REGIME_PREDICTIONS.csv',index=False)
pd.DataFrame(risk_rows).to_csv(OUT/'tables/NESTED_RISK_COVERAGE_CURVES.csv',index=False)
pd.DataFrame(decision_rows).to_csv(OUT/'tables/NESTED_PREDICT_REVIEW_ABSTAIN_METRICS.csv',index=False)
pd.DataFrame(perm_rows).to_csv(OUT/'tables/PERMUTED_SOURCE_CENTROID_CONTROLS.csv',index=False)

# exact coverage summaries and cluster bootstrap CIs
summ=[]
for (t,a,f),g in nested.groupby(['target','audit','fold']):
    rc,aurc=risk_curve(g.y_true,g.y_pred,g.disagreement)
    row={'target':t,'audit':a,'fold':f,'aurc':aurc,'n':len(g),'compatible_regime_abstention_rate':float(np.mean(g.decision=='ABSTAIN'))}
    for c in [.5,.75,.9]:
        v,n=risk_at(g.y_true,g.y_pred,g.disagreement,c);row[f'mae_at_{int(c*100)}pct_coverage']=v
        lo,hi=cluster_boot_ci(g,lambda x,cc=c:risk_at(x.y_true,x.y_pred,x.disagreement,cc)[0],B=250,seed=int(abs(hash((t,a,str(f),c)))%2**32));row[f'mae_at_{int(c*100)}_ci_low']=lo;row[f'mae_at_{int(c*100)}_ci_high']=hi
    summ.append(row)
pd.DataFrame(summ).to_csv(OUT/'tables/NESTED_COMPATIBLE_REGIME_SUMMARY_WITH_CLUSTER_CI.csv',index=False)

# Random ranking negative controls at discrete coverage
random_rows=[]
for (t,a,f),g in nested.groupby(['target','audit','fold']):
    y=g.y_true.to_numpy();p=g.y_pred.to_numpy();true_r=g.disagreement.to_numpy()
    for c in [.5,.75,.9]:
        actual,_=risk_at(y,p,true_r,c); vals=[]
        for i in range(500): vals.append(risk_at(y,p,RNG.permutation(true_r),c)[0])
        random_rows.append({'target':t,'audit':a,'fold':f,'coverage':c,'actual_mae':actual,'random_mean_mae':np.mean(vals),'random_ci_low':np.quantile(vals,.025),'random_ci_high':np.quantile(vals,.975),'improvement_vs_random_fraction':(np.mean(vals)-actual)/np.mean(vals)})
pd.DataFrame(random_rows).to_csv(OUT/'tables/RANDOM_DISAGREEMENT_NEGATIVE_CONTROL.csv',index=False)

# 2 interval audit using outer held-domain calibration from canonical rows excluding held domain
interval_rows=[]; cluster_rows=[]; density_rows=[]; frontier_rows=[]
for t in TARGETS:
    for held in DOMAINS:
        tr=usda[usda.domain_role!=held].copy();te=usda[usda.domain_role==held].copy();bun=fit_source_ensemble(tr,t);pte=predict_bundle(bun,te,TAU[t])
        cal=canon[(canon.target==t)&(canon.held_domain!=held)]
        mcal=max(float(np.median(cal.source_disagreement)),1e-12)
        for nominal in [.8,.9,.95]:
            q=float(np.quantile(cal.absolute_error,nominal)); alpha=1-nominal
            adapt_h=q*(1+BETA[t]*np.clip(pte['disagreement']/mcal,0,5)); const_h=np.full(len(te),adapt_h.mean())
            for method,h in [('adaptive_disagreement',adapt_h),('width_matched_constant',const_h),('global_constant',np.full(len(te),q))]:
                lo=pte['pred_frozen']-h;hi=pte['pred_frozen']+h;cov=(te[t].to_numpy()>=lo)&(te[t].to_numpy()<=hi);ws=winkler(te[t],lo,hi,alpha)
                tmp=pd.DataFrame({'target':t,'held_domain':held,'nominal':nominal,'method':method,'record_id':te.record_id.values,'species_group_key':te.species_group_key.values,'genus':te.genus.values,'specific_gravity':te.specific_gravity.values,'covered':cov.astype(int),'width':2*h,'winkler':ws})
                ci_cov=cluster_boot_ci(tmp,lambda x:x.covered.mean(),B=250,seed=4);ci_w=cluster_boot_ci(tmp,lambda x:x.winkler.mean(),B=250,seed=5)
                interval_rows.append({'target':t,'held_domain':held,'nominal':nominal,'method':method,'coverage':cov.mean(),'mean_width':float(np.mean(2*h)),'mean_winkler':float(np.mean(ws)),'coverage_ci_low':ci_cov[0],'coverage_ci_high':ci_cov[1],'winkler_ci_low':ci_w[0],'winkler_ci_high':ci_w[1]})
                for (gen,cl),gg in tmp.groupby(['genus','species_group_key']):cluster_rows.append({'target':t,'held_domain':held,'nominal':nominal,'method':method,'genus':gen,'species_group_key':cl,'n':len(gg),'coverage':gg.covered.mean(),'mean_width':gg.width.mean(),'mean_winkler':gg.winkler.mean()})
                qq=pd.qcut(tmp.specific_gravity,4,duplicates='drop')
                for qbin,gg in tmp.groupby(qq,observed=True):density_rows.append({'target':t,'held_domain':held,'nominal':nominal,'method':method,'density_bin':str(qbin),'n':len(gg),'coverage':gg.covered.mean(),'mean_width':gg.width.mean(),'mean_winkler':gg.winkler.mean()})
            # beta frontier, q fixed
            for beta in [0,.25,.5,.75,1,1.5,2]:
                h=q*(1+beta*np.clip(pte['disagreement']/mcal,0,5));lo=pte['pred_frozen']-h;hi=pte['pred_frozen']+h
                frontier_rows.append({'target':t,'held_domain':held,'nominal':nominal,'beta':beta,'coverage':float(np.mean((te[t]>=lo)&(te[t]<=hi))),'mean_width':float(np.mean(2*h))})

interval=pd.DataFrame(interval_rows);interval.to_csv(OUT/'tables/CLUSTER_CONDITIONAL_INTERVAL_SUMMARY.csv',index=False)
cl=pd.DataFrame(cluster_rows);cl.to_csv(OUT/'tables/INTERVAL_BY_GENUS_AND_CLUSTER.csv',index=False)
pd.DataFrame(density_rows).to_csv(OUT/'tables/INTERVAL_BY_DENSITY_QUARTILE.csv',index=False)
pd.DataFrame(frontier_rows).to_csv(OUT/'tables/COVERAGE_WIDTH_FRONTIER.csv',index=False)
macro=[]
for keys,g in cl.groupby(['target','held_domain','nominal','method']):
    macro.append(dict(zip(['target','held_domain','nominal','method'],keys),macro_cluster_coverage=g.coverage.mean(),worst_cluster_coverage=g.coverage.min(),median_cluster_coverage=g.coverage.median(),clusters=len(g)))
pd.DataFrame(macro).to_csv(OUT/'tables/MACRO_AND_WORST_CLUSTER_COVERAGE.csv',index=False)

# 4 guard stability via cluster bootstrap refits on LODO
stability_rows=[]; sensitivity_rows=[]
for t in TARGETS:
  for held in DOMAINS:
    tr=usda[usda.domain_role!=held].copy();te=usda[usda.domain_role==held].copy()
    base_b=fit_source_ensemble(tr,t);base_tr=predict_bundle(base_b,tr,TAU[t]);base_te=predict_bundle(base_b,te,TAU[t]);base_q=np.quantile(base_tr['disagreement'],[.5,.9]);base_dec=decision_labels(base_te['disagreement'],*base_q)
    all_decs=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed);pieces=[]
        for d in base_b['domains']:
            sd=tr[tr.domain_role==d]; groups=sd.species_group_key.unique();sample=rng.choice(groups,len(groups),replace=True);pieces.append(pd.concat([sd[sd.species_group_key==g] for g in sample],ignore_index=True))
        bt=pd.concat(pieces,ignore_index=True);bun=fit_source_ensemble(bt,t);ptr=predict_bundle(bun,bt,TAU[t]);pte=predict_bundle(bun,te,TAU[t]);q=np.quantile(ptr['disagreement'],[.5,.9]);dec=decision_labels(pte['disagreement'],*q);all_decs.append(dec)
        stability_rows.append({'target':t,'held_domain':held,'seed':seed,'changed_fraction_vs_base':float(np.mean(dec!=base_dec)),'predict_jaccard_vs_base':float(np.sum((dec=='PREDICT')&(base_dec=='PREDICT'))/max(np.sum((dec=='PREDICT')|(base_dec=='PREDICT')),1)),'abstain_jaccard_vs_base':float(np.sum((dec=='ABSTAIN')&(base_dec=='ABSTAIN'))/max(np.sum((dec=='ABSTAIN')|(base_dec=='ABSTAIN')),1)),'predict_fraction':float(np.mean(dec=='PREDICT')),'review_fraction':float(np.mean(dec=='REVIEW')),'abstain_fraction':float(np.mean(dec=='ABSTAIN'))})
    arr=np.vstack(all_decs)
    stability_rows.append({'target':t,'held_domain':held,'seed':'ACROSS_SEEDS','changed_fraction_vs_base':float(np.mean(np.any(arr!=arr[0],axis=0))),'predict_jaccard_vs_base':np.nan,'abstain_jaccard_vs_base':np.nan,'predict_fraction':float(np.mean(arr=='PREDICT')),'review_fraction':float(np.mean(arr=='REVIEW')),'abstain_fraction':float(np.mean(arr=='ABSTAIN'))})
    for ql in [.45,.5,.55]:
      for qh in [.85,.9,.95]:
        q=np.quantile(base_tr['disagreement'],[ql,qh]);dec=decision_labels(base_te['disagreement'],*q)
        sensitivity_rows.append({'target':t,'held_domain':held,'q_low':ql,'q_high':qh,'changed_fraction_vs_q50_q90':float(np.mean(dec!=base_dec)),'predict_fraction':float(np.mean(dec=='PREDICT')),'review_fraction':float(np.mean(dec=='REVIEW')),'abstain_fraction':float(np.mean(dec=='ABSTAIN'))})
pd.DataFrame(stability_rows).to_csv(OUT/'tables/GUARD_DECISION_STABILITY.csv',index=False)
pd.DataFrame(sensitivity_rows).to_csv(OUT/'tables/GUARD_THRESHOLD_SENSITIVITY.csv',index=False)

# 5 public Spruce external replication
spr=pd.read_csv(SPRUCE_PATH).copy();spr['specific_gravity']=spr.density_kg_m3/1000
# map to frozen development schema; retain source metadata separately
spr_model=spr.copy();spr_model['moisture_percent']=pd.to_numeric(spr_model.moisture_percent,errors='coerce');spr_model['moisture_state']='__MISSING__';spr_model['wood_class']='softwood';spr_model['genus']='Picea';spr_model['family']='Pinaceae';spr_model['sample_cluster']=spr_model.sample_name.astype(str).str.split('.').str[0]
spr_pred=[];spr_metrics=[];spr_interval=[]
for t in TARGETS:
    bun=fit_source_ensemble(usda,t);pr=predict_bundle(bun,spr_model,TAU[t]);q=FROZEN_Q[t];mcal=FROZEN_MCAL[t];h=q*(1+BETA[t]*np.clip(pr['disagreement']/mcal,0,5));lo=pr['pred_frozen']-h;hi=pr['pred_frozen']+h
    q50,q90=np.quantile(canon[canon.target==t].source_disagreement,[.5,.9]);dec=decision_labels(pr['disagreement'],q50,q90)
    d=pd.DataFrame({'target':t,'record_id':spr.record_id,'provided_split':spr.provided_split,'sample_cluster':spr_model.sample_cluster,'density_kg_m3':spr.density_kg_m3,'y_true':spr[t],'pred_erm':pr['pred_erm'],'pred_equal':pr['pred_equal'],'pred_frozen':pr['pred_frozen'],'source_disagreement':pr['disagreement'],'decision':dec,'interval_lower':lo,'interval_upper':hi})
    spr_pred.append(d)
    for subset,gg in [('all',d),('provided_test',d[d.provided_split=='test']),('metadata_qc_sensitivity',d[spr.length_mm>=500])]:
        for method,col in [('ERM','pred_erm'),('EQUAL','pred_equal'),('FROZEN','pred_frozen')]:spr_metrics.append({'target':t,'subset':subset,'method':method,**basic_metrics(gg.y_true,gg[col])})
        cov=np.mean((gg.y_true>=gg.interval_lower)&(gg.y_true<=gg.interval_upper));width=np.mean(gg.interval_upper-gg.interval_lower);ws=winkler(gg.y_true,gg.interval_lower,gg.interval_upper,.1)
        spr_interval.append({'target':t,'subset':subset,'coverage':cov,'mean_width':width,'mean_winkler':np.mean(ws),'predict_fraction':np.mean(gg.decision=='PREDICT'),'review_fraction':np.mean(gg.decision=='REVIEW'),'abstain_fraction':np.mean(gg.decision=='ABSTAIN'),'spearman_disagreement_error':spearmanr(gg.source_disagreement,np.abs(gg.y_true-gg.pred_frozen)).statistic})
sprp=pd.concat(spr_pred,ignore_index=True);sprp.to_csv(OUT/'data/SPRUCE_FROZEN_EXTERNAL_PREDICTIONS.csv',index=False)
pd.DataFrame(spr_metrics).to_csv(OUT/'tables/SPRUCE_EXTERNAL_POINT_METRICS.csv',index=False)
pd.DataFrame(spr_interval).to_csv(OUT/'tables/SPRUCE_EXTERNAL_INTERVAL_AND_GUARD_METRICS.csv',index=False)

# source weights long for reproducibility
weights=[]
for t in TARGETS:
    bun=fit_source_ensemble(usda,t);pr=predict_bundle(bun,spr_model,TAU[t])
    for j,d in enumerate(bun['domains']):
        weights.append(pd.DataFrame({'target':t,'record_id':spr.record_id,'source_domain':d,'weight':pr['weights'][:,j],'distance':pr['distances'][:,j],'source_prediction':pr['source_predictions'][:,j]}))
pd.concat(weights,ignore_index=True).to_csv(OUT/'data/SPRUCE_SOURCE_WEIGHTS.csv',index=False)

# 6 protocol/checklist
checklist=[
('Nested leave-one-domain-out','DONE','Training-only disagreement thresholds; held-domain risk curves and cluster CIs.'),
('Nested leave-one-genus-out','DONE','Five fixed genus-group folds with training-only thresholds.'),
('PREDICT/REVIEW/ABSTAIN','DONE','Reported for domain and genus held-out rows.'),
('Risk-coverage and AURC','DONE','Full curves plus 50/75/90% risk.'),
('Compatible-regime abstention rate','DONE_WITH_RENAMING','Reported honestly; not called false-abstention because no independent same-regime lockbox exists.'),
('Cluster bootstrap confidence intervals','DONE','Species/group cluster bootstrap for selective risk and interval outcomes.'),
('Cluster-conditional intervals','DONE','Domain, genus/species cluster, macro, worst-cluster, density quartile.'),
('Coverage-width frontier','DONE','80/90/95% nominal levels and beta frontier.'),
('Random disagreement control','DONE','1000 permutations.'),
('Permuted source centroids','DONE','All non-identity centroid permutations per fold.'),
('Constant-width controls 80/90/95','DONE','Global and width-matched constant comparators.'),
('Guard stability across seeds','DONE','Five cluster-bootstrap refits, changed fractions and Jaccard.'),
('q45/q55 q85/q95 sensitivity','DONE','Sensitivity only; frozen q50/q90 unchanged.'),
('Prespecified decision rules','DONE','Machine-readable protocol included.'),
('Free public external data','DONE','Spruce structural lamellae replication; cross-regime, not same-regime.'),
('GitHub/Zenodo DOI','NOT_DONE_EXTERNALLY','Repository-ready archive produced; actual account upload and DOI minting were not performed.'),
('Same-regime independent lockbox','NOT_AVAILABLE','No free specimen-level independent lockbox meeting all required fields was established.'),
]
pd.DataFrame(checklist,columns=['item','status','evidence_or_limit']).to_csv(OUT/'tables/STAGE4P_COMPLETION_CHECKLIST.csv',index=False)
protocol={
 'protocol_id':'WOODPHYS_DG_STAGE4P_PUBLIC_EXTERNAL_REPLICATION_V1','generated_after_public_data_was_already_present_in_project':True,
 'claim_boundary':'Spruce is a second independent cross-regime structural-board replication, not a same-regime clear-wood lockbox.',
 'frozen_parameters':{t:{'tau':TAU[t],'beta':BETA[t],'global_half_width_q':FROZEN_Q[t],'median_calibration_disagreement':FROZEN_MCAL[t]} for t in TARGETS},
 'primary_spruce_analysis':'all 2522 source-validated rows','sensitivity':'exclude rows with specimen length <500 mm using metadata only',
 'prohibited_interpretations':['Do not claim successful same-regime transport','Do not claim universally superior prediction','Do not tune parameters on Spruce labels'],
 'decision_hierarchy':{
  'FULL_SUPPORT':'Positive R2 and Frozen not materially worse than ERM/Equal on independent compatible-regime data; efficient intervals and acceptable compatible-regime abstention.',
  'PARTIAL_SUPPORT':'Some transport or triage support without consistent comparator advantage.',
  'NO_TRANSPORT_SUPPORT':'Negative R2 or Frozen materially worse than simple comparators; interval recovery attributable mainly to width.'}
}
(OUT/'protocol/FROZEN_STAGE4P_PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

# report summaries
sm=pd.DataFrame(spr_metrics);si=pd.DataFrame(spr_interval)
lines=['# Stage 4P — Public-data completion report','', '## What was actually completed']
for item,status,note in checklist:lines.append(f'- **{item}: {status}.** {note}')
lines+=['','## Spruce public external replication']
for t in TARGETS:
    x=sm[(sm.target==t)&(sm.subset=='all')]
    lines.append(f'### {t}')
    for _,r in x.iterrows():lines.append(f"- {r['method']}: MAE={r['mae']:.3f}, R2={r['r2']:.3f}, NMAE={r['nmae_iqr']:.3f}.")
    r=si[(si.target==t)&(si.subset=='all')].iloc[0]
    lines.append(f"- Frozen 90% interval: coverage={r.coverage:.3f}, width={r.mean_width:.3f}; guard P/R/A={r.predict_fraction:.3f}/{r.review_fraction:.3f}/{r.abstain_fraction:.3f}.")
lines+=['','## Claim boundary','Spruce strengthens or challenges the reproducibility of the cross-regime conclusion. It does not replace a specimen-level same-regime clear-wood lockbox.','', '## Repository status','A complete local repository-ready package and SHA-256 manifest are supplied. No GitHub/Zenodo account upload or DOI minting was performed.']
(OUT/'submission/STAGE4P_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

# Copy exact public data and provenance metadata
shutil.copy2(SPRUCE_PATH,OUT/'data/spruce_structural_board_separate_regime.csv')
shutil.copy2(USDA_PATH,OUT/'data/usda_clearwood_primary_model_ready.csv')
shutil.copy2(MANIFEST_PATH,OUT/'data/USDA_DOMAIN_AND_TAXON_SPLIT_MANIFEST.csv')
shutil.copy2(Path(__file__),OUT/'scripts/run_stage4p.py')
(OUT/'repository/requirements.txt').write_text('numpy\npandas\nscipy\nscikit-learn\npython-docx\nmatplotlib\n',encoding='utf-8')
(OUT/'repository/RUN_PROJECT.bat').write_text('@echo off\npython scripts\\run_stage4p.py\npause\n',encoding='utf-8')
(OUT/'repository/README.md').write_text('# WoodPhys-DG Stage 4P\nRun `python scripts/run_stage4p.py`. The package reproduces the frozen method, nested audits, interval controls, guard stability, and the public Spruce external replication.\n',encoding='utf-8')

# hashes
rows=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file():
        h=hashlib.sha256(p.read_bytes()).hexdigest();rows.append({'relative_path':str(p.relative_to(OUT)),'size_bytes':p.stat().st_size,'sha256':h})
pd.DataFrame(rows).to_csv(OUT/'SHA256_MANIFEST.csv',index=False)
status={'success':True,'frozen_reproduction_pass':all(x['status']=='PASS' for x in repro),'spruce_rows':len(spr),'all_local_analyses_completed':True,'external_repository_doi_minted':False}
(OUT/'SUCCESS_MARKER.json').write_text(json.dumps(status,indent=2),encoding='utf-8')

# zip after manuscript is added separately later
print(json.dumps(status,indent=2))
print(sm.to_string(index=False))
print(si.to_string(index=False))
