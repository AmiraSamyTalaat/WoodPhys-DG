from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
BASE = Path(__file__).resolve().parents[2]
DATA = BASE / 'data'
OUT = BASE / 'tables' / 'stage4r'
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [13, 42, 97, 131, 2026, 4099]
FULL = ['Density','Spacing','Pos','CA','MFA','DeltaMFA','DeltaDensity','Knots','SSDensity','RW','KnotA']
BASIC = ['Density','Spacing','Pos','CA']
DENSITY = ['Density']


def load_data() -> pd.DataFrame:
    d = pd.read_csv(DATA / 'Board_Data.csv').dropna(subset=['MOE','MOR','Tree']).copy()
    p = pd.read_csv(DATA / 'EXTERNAL_FROZEN_PREDICTIONS.csv')
    w = p.pivot(index='record_id', columns='target', values=['pred_frozen','source_disagreement']).reset_index()
    w.columns = ['BoardID'] + [f'{a}_{b}' for a, b in w.columns.tolist()[1:]]
    d = d.merge(w, on='BoardID', how='left', validate='one_to_one')
    d['Tree'] = d['Tree'].astype(str)
    return d.reset_index(drop=True)


def ridge() -> Pipeline:
    return Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('model', Ridge(alpha=10))])


def hgb(seed: int) -> Pipeline:
    return Pipeline([('impute', SimpleImputer(strategy='median')), ('model', HistGradientBoostingRegressor(max_iter=120, learning_rate=.05, max_leaf_nodes=12, min_samples_leaf=10, l2_regularization=3, random_state=seed))])


def extra(seed: int) -> Pipeline:
    return Pipeline([('impute', SimpleImputer(strategy='median')), ('model', ExtraTreesRegressor(n_estimators=120, max_depth=7, min_samples_leaf=4, max_features=.8, random_state=seed, n_jobs=-1))])


def metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float,float,float,float]:
    mae = mean_absolute_error(y, pred)
    rmse = mean_squared_error(y, pred) ** .5
    r2 = r2_score(y, pred)
    iqr = np.subtract(*np.quantile(y, [.75,.25]))
    return mae, rmse, r2, mae / iqr


def run_benchmark(d: pd.DataFrame) -> pd.DataFrame:
    rows, predictions = [], []
    for target in ['MOE','MOR']:
        frozen = f'pred_frozen_{target.lower()}_mpa'
        specs = [
            ('frozen_zero_shot', [frozen], None),
            ('density_ridge', DENSITY, 'ridge'),
            ('basic_hgb', BASIC, 'hgb'),
            ('full_hgb', FULL, 'hgb'),
            ('full_extra_trees', FULL, 'extra'),
            ('full_plus_frozen_hgb', FULL + [frozen], 'hgb'),
        ]
        for repeat, seed in enumerate(SEEDS):
            splitter = GroupKFold(5, shuffle=True, random_state=seed)
            for name, columns, kind in specs:
                pred = np.zeros(len(d)); fold_ids = np.zeros(len(d), dtype=int)
                for fold, (train, test) in enumerate(splitter.split(d, groups=d['Tree'])):
                    if kind is None:
                        fold_pred = d.iloc[test][frozen].to_numpy(float)
                    else:
                        model = ridge() if kind == 'ridge' else (hgb(seed + 10 * fold) if kind == 'hgb' else extra(seed + 10 * fold))
                        model.fit(d.iloc[train][columns], d.iloc[train][target])
                        fold_pred = model.predict(d.iloc[test][columns])
                    pred[test] = fold_pred; fold_ids[test] = fold
                mae, rmse, r2, nmae = metrics(d[target].to_numpy(), pred)
                rows.append([target, repeat, seed, name, mae, rmse, r2, nmae])
                for i in range(len(d)):
                    predictions.append([target, repeat, seed, fold_ids[i], d.iloc[i]['BoardID'], d.iloc[i]['Tree'], d.iloc[i][target], name, pred[i], abs(d.iloc[i][target] - pred[i])])
    raw = pd.DataFrame(rows, columns=['target','repeat','seed','model','mae','rmse','r2','nmae'])
    raw.to_csv(OUT / 'PINUS_MODEL_BENCHMARK_RAW.csv', index=False)
    summary = raw.groupby(['target','model']).agg(
        median_mae=('mae','median'), q25_mae=('mae',lambda x:x.quantile(.25)), q75_mae=('mae',lambda x:x.quantile(.75)),
        median_r2=('r2','median'), min_r2=('r2','min'), max_r2=('r2','max'), median_nmae=('nmae','median')).reset_index()
    summary.to_csv(OUT / 'PINUS_MODEL_BENCHMARK_SUMMARY.csv', index=False)
    pd.DataFrame(predictions, columns=['target','repeat','seed','fold','board_id','tree','y_true','model','prediction','abs_error']).to_csv(OUT / 'PINUS_MODEL_PREDICTIONS.csv', index=False)
    return summary


def run_label_efficiency(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ['MOE','MOR']:
        for fraction in [.2,.4,.6,.8]:
            for repeat in range(8):
                train, test = next(GroupShuffleSplit(n_splits=1, train_size=fraction, random_state=800+repeat).split(d, groups=d['Tree']))
                for name, columns, kind in [('density_ridge',DENSITY,'ridge'),('full_hgb',FULL,'hgb'),('full_extra_trees',FULL,'extra')]:
                    model = ridge() if kind == 'ridge' else (hgb(900+repeat) if kind == 'hgb' else extra(900+repeat))
                    model.fit(d.iloc[train][columns], d.iloc[train][target])
                    pred = model.predict(d.iloc[test][columns])
                    mae, rmse, r2, nmae = metrics(d.iloc[test][target].to_numpy(), pred)
                    rows.append([target,fraction,repeat,name,len(train),d.iloc[train]['Tree'].nunique(),len(test),d.iloc[test]['Tree'].nunique(),mae,rmse,r2,nmae])
    raw = pd.DataFrame(rows, columns=['target','tree_fraction','repeat','model','n_train','n_train_trees','n_test','n_test_trees','mae','rmse','r2','nmae'])
    raw.to_csv(OUT / 'PINUS_LABEL_EFFICIENCY_RAW.csv', index=False)
    summary = raw.groupby(['target','tree_fraction','model']).agg(
        median_r2=('r2','median'), q25_r2=('r2',lambda x:x.quantile(.25)), q75_r2=('r2',lambda x:x.quantile(.75)),
        positive_r2_fraction=('r2',lambda x:np.mean(np.asarray(x)>0)), median_mae=('mae','median'), median_train_trees=('n_train_trees','median')).reset_index()
    summary.to_csv(OUT / 'PINUS_LABEL_EFFICIENCY_SUMMARY.csv', index=False)
    return summary


if __name__ == '__main__':
    data = load_data()
    benchmark = run_benchmark(data)
    label = run_label_efficiency(data)
    print(benchmark.sort_values(['target','median_r2'], ascending=[True,False]).to_string(index=False))
    print(label.to_string(index=False))
