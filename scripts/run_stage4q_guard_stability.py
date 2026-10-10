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
all_dec={};risk_rows=[]
for target in ['moe_mpa','mor_mpa']:
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
  all_dec[(target,seed)]=dec
  risk_rows.append([target,seed,q50,q90,float(np.mean(dec=='PREDICT')),float(np.mean(dec=='REVIEW')),float(np.mean(dec=='ABSTAIN'))])
# pairwise stability
rows=[]
for target in ['moe_mpa','mor_mpa']:
 for s1,s2 in itertools.combinations(range(6),2):
  a=all_dec[(target,s1)];b=all_dec[(target,s2)]
  changed=np.mean(a!=b)
  vals=[target,s1,s2,changed]
  for lab in ['PREDICT','REVIEW','ABSTAIN']:
   A=a==lab;B2=b==lab;union=np.sum(A|B2);jac=np.sum(A&B2)/union if union else 1
   vals.append(jac)
  rows.append(vals)
pair=pd.DataFrame(rows,columns=['target','seed_a','seed_b','changed_fraction','jaccard_predict','jaccard_review','jaccard_abstain'])
pair.to_csv(B/'tables/STAGE4Q_LEARNED_GUARD_PAIRWISE_STABILITY.csv',index=False)
sumr=pair.groupby('target').agg(mean_changed=('changed_fraction','mean'),max_changed=('changed_fraction','max'),mean_jaccard_predict=('jaccard_predict','mean'),mean_jaccard_review=('jaccard_review','mean'),mean_jaccard_abstain=('jaccard_abstain','mean')).reset_index()
sumr.to_csv(B/'tables/STAGE4Q_LEARNED_GUARD_STABILITY_SUMMARY.csv',index=False)
pd.DataFrame(risk_rows,columns=['target','seed','q50','q90','predict_fraction','review_fraction','abstain_fraction']).to_csv(B/'tables/STAGE4Q_LEARNED_GUARD_THRESHOLDS.csv',index=False)
print(sumr.to_string(index=False))
