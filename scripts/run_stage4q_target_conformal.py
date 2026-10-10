import pandas as pd,numpy as np,warnings
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
warnings.filterwarnings('ignore')
B=Path(__file__).resolve().parents[1]
df=pd.read_csv(B/'data/spruce_structural_board_separate_regime.csv');df['sample_cluster']=df.sample_name.astype(str).str.split('.').str[0]
features=['density_kg_m3','length_mm','width_mm','height_mm','weight_g','quality_face_1','quality_face_2']
def base(seed):return HistGradientBoostingRegressor(max_iter=160,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=15,l2_regularization=2,random_state=seed)
def risk(seed):return ExtraTreesRegressor(n_estimators=120,max_depth=10,min_samples_leaf=15,max_features=.8,random_state=seed,n_jobs=-1)
def winkler(y,lo,hi,alpha):
 w=hi-lo; return np.where(y<lo,w+(2/alpha)*(lo-y),np.where(y>hi,w+(2/alpha)*(y-hi),w))
rows=[]
for target in ['moe_mpa','mor_mpa']:
 for ofold,(a,b) in enumerate(GroupKFold(5).split(df,groups=df.sample_cluster)):
  A=df.iloc[a].copy().reset_index(drop=True);Btest=df.iloc[b].copy().reset_index(drop=True)
  # base OOF residuals on A
  A['oof_pred']=np.nan
  inner=GroupKFold(4)
  splits=list(inner.split(A,groups=A.sample_cluster))
  for ifold,(ia,ib) in enumerate(splits):
   m=base(1000+ofold*10+ifold);m.fit(A.iloc[ia][features],A.iloc[ia][target]);A.loc[ib,'oof_pred']=m.predict(A.iloc[ib][features])
  A['absres']=(A[target]-A.oof_pred).abs()
  # cross-fitted risk predictions for honest normalization
  A['risk_oof']=np.nan
  for ifold,(ia,ib) in enumerate(splits):
   r=risk(2000+ofold*10+ifold);r.fit(A.iloc[ia][features],np.log1p(A.iloc[ia].absres));A.loc[ib,'risk_oof']=np.maximum(1e-6,np.expm1(r.predict(A.iloc[ib][features])))
  # final base and risk
  m=base(3000+ofold);m.fit(A[features],A[target]);pred=m.predict(Btest[features])
  r=risk(4000+ofold);r.fit(A[features],np.log1p(A.absres));risk_test=np.maximum(1e-6,np.expm1(r.predict(Btest[features])))
  for nominal in [.80,.90,.95]:
   alpha=1-nominal
   qconst=np.quantile(A.absres,nominal,method='higher')
   qnorm=np.quantile(A.absres/np.maximum(A.risk_oof,1e-6),nominal,method='higher')
   for method,half in [('constant',np.full(len(Btest),qconst)),('adaptive_crossfit_risk',qnorm*risk_test)]:
    lo=pred-half;hi=pred+half;y=Btest[target].values
    rows.append([target,ofold,nominal,method,float(np.mean((y>=lo)&(y<=hi))),float(np.mean(hi-lo)),float(np.mean(winkler(y,lo,hi,alpha)))])
out=pd.DataFrame(rows,columns=['target','outer_fold','nominal','interval_method','coverage','mean_width','mean_winkler'])
out.to_csv(B/'tables/STAGE4Q_TARGET_CONFORMAL_FOLD_RESULTS.csv',index=False)
s=out.groupby(['target','nominal','interval_method']).agg(mean_coverage=('coverage','mean'),sd_coverage=('coverage','std'),mean_width=('mean_width','mean'),mean_winkler=('mean_winkler','mean')).reset_index()
s.to_csv(B/'tables/STAGE4Q_TARGET_CONFORMAL_SUMMARY.csv',index=False)
print(s.to_string(index=False))
