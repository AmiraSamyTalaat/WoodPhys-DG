import pandas as pd,numpy as np,warnings
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from catboost import CatBoostRegressor
warnings.filterwarnings('ignore')
B=Path(__file__).resolve().parents[1]
df=pd.read_csv(B/'data/spruce_structural_board_separate_regime.csv'); df['sample_cluster']=df.sample_name.astype(str).str.split('.').str[0]
features=['density_kg_m3','length_mm','width_mm','height_mm','weight_g','quality_face_1','quality_face_2']
def metrics(y,p):return mean_absolute_error(y,p),mean_squared_error(y,p)**.5,r2_score(y,p)
def cat(seed): return CatBoostRegressor(iterations=250,depth=6,learning_rate=.05,l2_leaf_reg=5,loss_function='RMSE',random_seed=seed,verbose=False,thread_count=4)
def rf(seed):return RandomForestRegressor(n_estimators=250,min_samples_leaf=5,max_features=.8,random_state=seed,n_jobs=-1)
rows=[]
for target in ['moe_mpa','mor_mpa']:
 tr=df[df.provided_split=='train'];te=df[df.provided_split=='test']
 for name,fn in [('catboost',cat),('random_forest',rf)]:
  m=fn(42);m.fit(tr[features],tr[target]);p=m.predict(te[features]);rows.append([target,'provided_test',name,*metrics(te[target],p)])
  po=np.zeros(len(df))
  for fold,(a,b) in enumerate(GroupKFold(5).split(df,groups=df.sample_cluster)):
   m=fn(100+fold);m.fit(df.iloc[a][features],df.iloc[a][target]);po[b]=m.predict(df.iloc[b][features])
  rows.append([target,'group5_oof',name,*metrics(df[target],po)])
out=pd.DataFrame(rows,columns=['target','evaluation','model','mae','rmse','r2'])
out.to_csv(B/'tables/STAGE4Q_EXTRA_MODEL_BENCHMARK.csv',index=False)
print(out.to_string(index=False))
