import pandas as pd,numpy as np,warnings,itertools
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
warnings.filterwarnings('ignore')
B=Path(__file__).resolve().parents[1]
df=pd.read_csv(B/'data/spruce_structural_board_separate_regime.csv');df['sample_cluster']=df.sample_name.astype(str).str.split('.').str[0]
features=['density_kg_m3','length_mm','width_mm','height_mm','weight_g','quality_face_1','quality_face_2']
def base(seed):return HistGradientBoostingRegressor(max_iter=140,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=15,l2_regularization=2,random_state=seed)
def risk(seed):return ExtraTreesRegressor(n_estimators=100,max_depth=10,min_samples_leaf=15,max_features=.8,random_state=seed,n_jobs=-1)
raw=[]; summary=[]; stability=[]
for target in ['moe_mpa','mor_mpa']:
 decs=[]; errs=[]; preds=[]; risks=[]
 for seed in range(6):
  splits=list(GroupKFold(5,shuffle=True,random_state=500+seed).split(df,groups=df.sample_cluster))
  oof=np.zeros(len(df))
  for k,(a,b) in enumerate(splits):
   m=base(600+seed*10+k);m.fit(df.iloc[a][features],df.iloc[a][target]);oof[b]=m.predict(df.iloc[b][features])
  err=np.abs(df[target].values-oof)
  rp=np.zeros(len(df))
  for k,(a,b) in enumerate(splits):
   r=risk(700+seed*10+k);r.fit(df.iloc[a][features],np.log1p(err[a]));rp[b]=np.maximum(1e-6,np.expm1(r.predict(df.iloc[b][features])))
  q50,q90=np.quantile(rp,[.5,.9]);dec=np.where(rp<=q50,'PREDICT',np.where(rp<=q90,'REVIEW','ABSTAIN'))
  decs.append(dec);errs.append(err);preds.append(oof);risks.append(rp)
 D=np.array(decs); E=np.array(errs); P=np.array(preds); R=np.array(risks)
 # strict 5/6 consensus; uncertainty routed to review
 pc=(D=='PREDICT').sum(0);ac=(D=='ABSTAIN').sum(0)
 cons=np.where(pc>=5,'PREDICT',np.where(ac>=5,'ABSTAIN','REVIEW'))
 meanerr=E.mean(0); meanpred=P.mean(0); meanrisk=R.mean(0)
 for i,row in df.reset_index(drop=True).iterrows():raw.append([target,row.record_id,row[target],meanpred[i],meanerr[i],meanrisk[i],cons[i],pc[i],ac[i]])
 overall=meanerr.mean()
 for lab in ['PREDICT','REVIEW','ABSTAIN']:
  mask=cons==lab; summary.append([target,lab,mask.mean(),mask.sum(),meanerr[mask].mean() if mask.any() else np.nan,(meanerr[mask].mean()/overall if mask.any() else np.nan)])
 # leave-one-replicate-out consensus (4/5 threshold), audit stability of consensus mechanism
 loo=[]
 for omit in range(6):
  X=np.delete(D,omit,axis=0);pc2=(X=='PREDICT').sum(0);ac2=(X=='ABSTAIN').sum(0);loo.append(np.where(pc2>=4,'PREDICT',np.where(ac2>=4,'ABSTAIN','REVIEW')))
 for a,b in itertools.combinations(range(6),2):
  A=loo[a];B2=loo[b];vals=[target,a,b,np.mean(A!=B2)]
  for lab in ['PREDICT','REVIEW','ABSTAIN']:
   x=A==lab;y=B2==lab;u=(x|y).sum();vals.append((x&y).sum()/u if u else 1)
  stability.append(vals)
pd.DataFrame(raw,columns=['target','record_id','y_true','ensemble_oof_prediction','mean_abs_error','ensemble_risk','consensus_decision','predict_votes','abstain_votes']).to_csv(B/'tables/STAGE4Q_CONSENSUS_GUARD_PREDICTIONS.csv',index=False)
s=pd.DataFrame(summary,columns=['target','decision','fraction','n','mean_abs_error','error_ratio_to_overall']);s.to_csv(B/'tables/STAGE4Q_CONSENSUS_GUARD_PERFORMANCE.csv',index=False)
st=pd.DataFrame(stability,columns=['target','omit_a','omit_b','changed_fraction','jaccard_predict','jaccard_review','jaccard_abstain']);st.to_csv(B/'tables/STAGE4Q_CONSENSUS_GUARD_LOO_STABILITY.csv',index=False)
ss=st.groupby('target').agg(mean_changed=('changed_fraction','mean'),max_changed=('changed_fraction','max'),mean_jaccard_predict=('jaccard_predict','mean'),mean_jaccard_review=('jaccard_review','mean'),mean_jaccard_abstain=('jaccard_abstain','mean')).reset_index();ss.to_csv(B/'tables/STAGE4Q_CONSENSUS_GUARD_STABILITY_SUMMARY.csv',index=False)
print('PERFORMANCE\n',s.to_string(index=False));print('\nSTABILITY\n',ss.to_string(index=False))
