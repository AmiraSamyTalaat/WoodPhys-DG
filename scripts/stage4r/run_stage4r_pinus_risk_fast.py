from __future__ import annotations
import math, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')
B=Path(__file__).resolve().parents[2];O=B/'tables'/'stage4r';O.mkdir(exist_ok=True,parents=True)
F=['Density','Spacing','Pos','CA','MFA','DeltaMFA','DeltaDensity','Knots','SSDensity','RW','KnotA'];SEEDS=[13,42,97]
def model(seed,leaf=4):return Pipeline([('i',SimpleImputer(strategy='median')),('m',ExtraTreesRegressor(n_estimators=90,max_depth=7,min_samples_leaf=leaf,max_features=.8,random_state=seed,n_jobs=-1))])
def fq(v,n):v=np.sort(np.asarray(v,float));return v[min(len(v),math.ceil((len(v)+1)*n))-1]
def wink(y,l,h,a):w=h-l;return np.where(y<l,w+2/a*(l-y),np.where(y>h,w+2/a*(y-h),w))
d=pd.read_csv(B/'data/Board_Data.csv').dropna(subset=['MOE','MOR','Tree']).copy();d.Tree=d.Tree.astype(str)
riskrows=[];confrows=[];decrows=[]
for target in ['MOE','MOR']:
 for rep,seed in enumerate(SEEDS):
  outer=GroupKFold(5,shuffle=True,random_state=seed)
  for fold,(a,b) in enumerate(outer.split(d,groups=d.Tree)):
   A=d.iloc[a].copy().reset_index(drop=True);T=d.iloc[b].copy().reset_index(drop=True)
   inner=GroupKFold(4,shuffle=True,random_state=seed+fold+100);spl=list(inner.split(A,groups=A.Tree));A['base_oof']=np.nan
   for k,(ia,ib) in enumerate(spl):
    m=model(seed+1000+k);m.fit(A.iloc[ia][F],A.iloc[ia][target]);A.loc[ib,'base_oof']=m.predict(A.iloc[ib][F])
   A['err']=(A[target]-A.base_oof).abs();rf=F+['base_oof'];A['risk_oof']=np.nan
   for k,(ia,ib) in enumerate(spl):
    r=model(seed+2000+k,leaf=5);r.fit(A.iloc[ia][rf],np.log1p(A.iloc[ia].err));A.loc[ib,'risk_oof']=np.maximum(1e-6,np.expm1(r.predict(A.iloc[ib][rf])))
   m=model(seed+3000+fold);m.fit(A[F],A[target]);pred=m.predict(T[F]);T['base_oof']=pred
   r=model(seed+4000+fold,leaf=5);r.fit(A[rf],np.log1p(A.err));risk=np.maximum(1e-6,np.expm1(r.predict(T[rf])));err=np.abs(T[target].to_numpy()-pred)
   q50,q90=np.quantile(A.risk_oof,[.5,.9]);dec=np.where(risk<=q50,'PREDICT',np.where(risk<=q90,'REVIEW','ABSTAIN'))
   rho=float(spearmanr(risk,err).statistic);order=np.argsort(risk);grid=np.linspace(.1,1,19);sel=[err[order[:max(1,int(np.ceil(c*len(T))))]].mean() for c in grid];aurc=float(np.trapezoid(sel,grid)/err.mean());hi=err>=np.quantile(err,.75);rh=risk>=np.quantile(risk,.75);cap=float((hi&rh).sum()/max(1,hi.sum()));riskrows.append([target,rep,fold,rho,aurc,cap])
   for i in range(len(T)):decrows.append([target,rep,fold,T.iloc[i].BoardID,T.iloc[i].Tree,T.iloc[i][target],pred[i],err[i],risk[i],dec[i]])
   for nom in [.8,.9,.95]:
    qc=fq(A.err,nom);qa=fq(A.err/np.maximum(A.risk_oof,1e-6),nom)
    for name,half in [('constant_crossfit',np.full(len(T),qc)),('adaptive_learned_risk',qa*risk)]:
     lo=pred-half;up=pred+half;y=T[target].to_numpy();cov=(y>=lo)&(y<=up);tc=pd.DataFrame({'tree':T.Tree,'covered':cov}).groupby('tree').covered.mean();confrows.append([target,rep,fold,nom,name,cov.mean(),tc.mean(),tc.min(),np.mean(2*half),np.mean(wink(y,lo,up,1-nom))])
r=pd.DataFrame(riskrows,columns=['target','repeat','fold','spearman','normalized_aurc','top25_capture']);r.to_csv(O/'PINUS_RISK_FOLDS.csv',index=False);rs=r.groupby('target').agg(mean_spearman=('spearman','mean'),sd_spearman=('spearman','std'),mean_normalized_aurc=('normalized_aurc','mean'),mean_top25_capture=('top25_capture','mean')).reset_index();rs.to_csv(O/'PINUS_RISK_SUMMARY.csv',index=False)
c=pd.DataFrame(confrows,columns=['target','repeat','fold','nominal','method','record_coverage','macro_tree_coverage','worst_tree_coverage','mean_width','mean_winkler']);c.to_csv(O/'PINUS_CONFORMAL_FOLDS.csv',index=False);cs=c.groupby(['target','nominal','method']).agg(mean_record_coverage=('record_coverage','mean'),mean_macro_tree_coverage=('macro_tree_coverage','mean'),mean_worst_tree_coverage=('worst_tree_coverage','mean'),mean_width=('mean_width','mean'),mean_winkler=('mean_winkler','mean')).reset_index();cs.to_csv(O/'PINUS_CONFORMAL_SUMMARY.csv',index=False)
dec=pd.DataFrame(decrows,columns=['target','repeat','fold','board_id','tree','y_true','prediction','abs_error','risk','decision']);dec.to_csv(O/'PINUS_GUARD_DECISIONS.csv',index=False)
# 2/3 consensus; ambiguous routed to review
piv=dec.pivot_table(index=['target','board_id','tree','y_true'],columns='repeat',values='decision',aggfunc='first');err=dec.groupby(['target','board_id','tree','y_true']).abs_error.mean();rows=[]
for idx,x in piv.iterrows():
 v=x.dropna().tolist();pv=sum(z=='PREDICT' for z in v);av=sum(z=='ABSTAIN' for z in v);lab='PREDICT' if pv>=2 else ('ABSTAIN' if av>=2 else 'REVIEW');rows.append([*idx,err.loc[idx],lab,pv,av])
co=pd.DataFrame(rows,columns=['target','board_id','tree','y_true','mean_abs_error','consensus_decision','predict_votes','abstain_votes']);co.to_csv(O/'PINUS_CONSENSUS_PREDICTIONS.csv',index=False);perf=[]
for t,g in co.groupby('target'):
 base=g.mean_abs_error.mean()
 for lab in ['PREDICT','REVIEW','ABSTAIN']:
  x=g[g.consensus_decision==lab];perf.append([t,lab,len(x),len(x)/len(g),x.mean_abs_error.mean() if len(x) else np.nan,x.mean_abs_error.mean()/base if len(x) else np.nan])
pd.DataFrame(perf,columns=['target','decision','n','fraction','mean_abs_error','error_ratio']).to_csv(O/'PINUS_CONSENSUS_PERFORMANCE.csv',index=False)
print('RISK\n',rs.to_string(index=False));print('\nCONF90\n',cs[cs.nominal==.9].to_string(index=False));print('\nGUARD\n',pd.DataFrame(perf,columns=['target','decision','n','fraction','mean_abs_error','error_ratio']).to_string(index=False))
