import pandas as pd, numpy as np, warnings
from pathlib import Path
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
BASE=Path(__file__).resolve().parents[1]
df=pd.read_csv(BASE/'data/spruce_structural_board_separate_regime.csv')
preds=pd.read_csv(BASE/'data/SPRUCE_FROZEN_EXTERNAL_PREDICTIONS.csv')
# long to wide
wide=preds.pivot(index='record_id',columns='target',values=['pred_frozen','source_disagreement']).reset_index()
wide.columns=['record_id']+[f'{a}_{b}' for a,b in wide.columns.tolist()[1:]]
df=df.merge(wide,on='record_id',how='left')
df['sample_cluster']=df['sample_name'].astype(str).str.split('.').str[0]
features=['density_kg_m3','length_mm','width_mm','height_mm','weight_g','quality_face_1','quality_face_2']
def hgb(seed=42): return HistGradientBoostingRegressor(max_iter=160,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=15,l2_regularization=2,random_state=seed)
def met(y,p): return mean_absolute_error(y,p), mean_squared_error(y,p)**.5, r2_score(y,p)
# Benchmark official split + GroupKFold
bench=[]; oof_store={}
for target in ['moe_mpa','mor_mpa']:
  fp=f'pred_frozen_{target}'
  tr=df[df.provided_split=='train']; te=df[df.provided_split=='test']
  specs=[('density_ridge', ['density_kg_m3'], 'direct'),('target_hgb',features,'direct'),('frozen_residual_hgb',features,'residual'),('stacked_hgb',features+[fp],'direct')]
  for name,cols,kind in specs:
    m=Ridge(alpha=10) if name=='density_ridge' else hgb(42)
    yy=tr[target]-tr[fp] if kind=='residual' else tr[target]
    m.fit(tr[cols],yy); p=m.predict(te[cols]);
    if kind=='residual': p=te[fp].values+p
    ma,rm,r2=met(te[target],p); bench.append([target,'provided_test',name,len(tr),len(te),ma,rm,r2])
  # frozen baseline
  p=te[fp].values; ma,rm,r2=met(te[target],p); bench.append([target,'provided_test','frozen_zero_shot',0,len(te),ma,rm,r2])
  gkf=GroupKFold(5)
  for name,cols,kind in specs[1:]:
    po=np.zeros(len(df))
    for fold,(a,b) in enumerate(gkf.split(df,groups=df.sample_cluster)):
      A=df.iloc[a]; B=df.iloc[b]; m=hgb(100+fold)
      yy=A[target]-A[fp] if kind=='residual' else A[target]
      m.fit(A[cols],yy); pp=m.predict(B[cols]);
      if kind=='residual': pp=B[fp].values+pp
      po[b]=pp
    ma,rm,r2=met(df[target],po); bench.append([target,'group5_oof',name,len(df),len(df),ma,rm,r2]); oof_store[(target,name)]=po
bench=pd.DataFrame(bench,columns=['target','evaluation','model','n_train','n_test','mae','rmse','r2'])
bench.to_csv(BASE/'tables/STAGE4Q_TARGET_ADAPTATION_BENCHMARK.csv',index=False)
print('BENCHMARK\n',bench.to_string(index=False))
# Label efficiency, cluster-separated, 4 repetitions
le=[]
for target in ['moe_mpa','mor_mpa']:
 fp=f'pred_frozen_{target}'
 for frac in [.05,.10,.20,.40,.60,.80]:
  for rep in range(4):
   a,b=next(GroupShuffleSplit(n_splits=1,train_size=frac,random_state=700+rep).split(df,groups=df.sample_cluster))
   A=df.iloc[a]; B=df.iloc[b]
   for name,cols,kind in [('target_hgb',features,'direct'),('frozen_residual_hgb',features,'residual'),('stacked_hgb',features+[fp],'direct')]:
    m=hgb(800+rep); yy=A[target]-A[fp] if kind=='residual' else A[target]; m.fit(A[cols],yy); pp=m.predict(B[cols]);
    if kind=='residual': pp=B[fp].values+pp
    ma,rm,r2=met(B[target],pp); le.append([target,frac,rep,name,len(A),A.sample_cluster.nunique(),len(B),ma,rm,r2])
le=pd.DataFrame(le,columns=['target','label_fraction','repeat','model','n_train','n_train_clusters','n_test','mae','rmse','r2'])
le.to_csv(BASE/'tables/STAGE4Q_LABEL_EFFICIENCY_RAW.csv',index=False)
sumle=le.groupby(['target','label_fraction','model']).agg(median_r2=('r2','median'),q25_r2=('r2',lambda x:x.quantile(.25)),q75_r2=('r2',lambda x:x.quantile(.75)),median_mae=('mae','median'),median_n_train=('n_train','median'),median_clusters=('n_train_clusters','median')).reset_index()
sumle.to_csv(BASE/'tables/STAGE4Q_LABEL_EFFICIENCY_SUMMARY.csv',index=False)
print('\nLABEL\n',sumle.to_string(index=False))
# Nested learned residual risk; fully group-separated
risk_rows=[]; risk_pred=[]
for target in ['moe_mpa','mor_mpa']:
  sd=f'source_disagreement_{target}'
  outer=GroupKFold(5)
  for fold,(a,b) in enumerate(outer.split(df,groups=df.sample_cluster)):
    A=df.iloc[a].copy(); B=df.iloc[b].copy()
    # inner OOF residuals
    A['_oof']=np.nan
    for ia,ib in GroupKFold(4).split(A,groups=A.sample_cluster):
      m=hgb(900+fold); m.fit(A.iloc[ia][features],A.iloc[ia][target]); A.iloc[ib,A.columns.get_loc('_oof')]=m.predict(A.iloc[ib][features])
    A['_err']=(A[target]-A['_oof']).abs()
    base=hgb(1000+fold); base.fit(A[features],A[target]); pp=base.predict(B[features]); err=np.abs(B[target].values-pp)
    # stable residual-risk model
    rm=ExtraTreesRegressor(n_estimators=120,max_depth=10,min_samples_leaf=15,max_features=.8,random_state=1100+fold,n_jobs=-1)
    rm.fit(A[features],np.log1p(A['_err'])); learned=np.maximum(1e-6,np.expm1(rm.predict(B[features])))
    # thresholds based only on training learned risk
    trainrisk=np.maximum(1e-6,np.expm1(rm.predict(A[features]))); q50,q90=np.quantile(trainrisk,[.5,.9])
    decision=np.where(learned<=q50,'PREDICT',np.where(learned<=q90,'REVIEW','ABSTAIN'))
    for score_name,score in [('source_disagreement',B[sd].values),('learned_target_residual_risk',learned)]:
      rho=float(spearmanr(score,err).statistic)
      order=np.argsort(score); cov=np.linspace(.05,1,20); selective=[]
      for c in cov:
        k=max(1,int(c*len(B))); selective.append(float(err[order[:k]].mean()))
      aurc=float(np.trapz(selective,cov)/err.mean())
      actual_hi=err>=np.quantile(err,.75); score_hi=score>=np.quantile(score,.75); capture=float((actual_hi&score_hi).sum()/actual_hi.sum())
      risk_rows.append([target,fold,score_name,rho,aurc,capture])
    for i,row in B.reset_index(drop=True).iterrows():
      risk_pred.append([target,fold,row.record_id,row[target],pp[i],err[i],row[sd],learned[i],decision[i]])
risk=pd.DataFrame(risk_rows,columns=['target','fold','risk_score','spearman','normalized_aurc','top25_error_capture'])
risk.to_csv(BASE/'tables/STAGE4Q_RISK_SCORE_FOLD_METRICS.csv',index=False)
risksum=risk.groupby(['target','risk_score']).agg(mean_spearman=('spearman','mean'),sd_spearman=('spearman','std'),mean_normalized_aurc=('normalized_aurc','mean'),mean_top25_capture=('top25_error_capture','mean')).reset_index()
risksum.to_csv(BASE/'tables/STAGE4Q_RISK_SCORE_SUMMARY.csv',index=False)
pd.DataFrame(risk_pred,columns=['target','fold','record_id','y_true','prediction','abs_error','source_disagreement','learned_target_residual_risk','decision']).to_csv(BASE/'tables/STAGE4Q_RISK_PREDICTIONS.csv',index=False)
print('\nRISK\n',risksum.to_string(index=False))
