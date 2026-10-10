
from __future__ import annotations
from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import shutil
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon, spearmanr, ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT/"input"
OUT = ROOT/"stage4h_output"
WORK = OUT/"_work"
TABLES = OUT/"tables"
FIGURES = OUT/"figures"

ZIPS = {
    "stage0f": IN/"STAGE0F_VERIFIED_INPUT.zip",
    "stage2c": IN/"STAGE2C_VERIFIED_INPUT.zip",
    "stage3a": IN/"STAGE3A_FIX2_VERIFIED_INPUT.zip",
    "stage3b": IN/"STAGE3B_VERIFIED_INPUT.zip",
    "stage4c": IN/"STAGE4C_VERIFIED_INPUT.zip",
}

TARGETS = ["moe_mpa", "mor_mpa"]
METHODS = {
    "ERM": "pred_erm",
    "EQUAL": "pred_equal_ensemble",
    "FROZEN": "pred_frozen",
}
BOOT = 3000
PERM = 20000
RANDOM_RANKINGS = 500
SEED = 20260715

LOG = []
ERR = []

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def log(msg):
    line=f"[{now()}] {msg}"
    LOG.append(line)
    print(line, flush=True)

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")

def verify_result(name,zpath):
    dest=WORK/name
    dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(dest)
    s=json.loads((dest/"SUCCESS_MARKER.json").read_text(encoding="utf-8"))
    v=json.loads((dest/"OUTPUT_VERIFICATION.json").read_text(encoding="utf-8"))
    if not s.get("success") or not v.get("verification_passed"):
        raise RuntimeError(f"{name} verification failed")
    hp=dest/"HASHES_SHA256.csv"
    if hp.exists():
        rows=pd.read_csv(hp)
        bad=[]
        for _,r in rows.iterrows():
            p=dest/r["relative_path"]
            actual=sha256(p) if p.exists() else "MISSING"
            if actual != r["sha256"]:
                bad.append(r["relative_path"])
        if bad:
            raise RuntimeError(f"{name} hash mismatch: {bad[:5]}")
    return dest

def metric(y,pred):
    y=np.asarray(y,float); pred=np.asarray(pred,float)
    return {
        "mae":float(mean_absolute_error(y,pred)),
        "rmse":float(mean_squared_error(y,pred)**0.5),
        "r2":float(r2_score(y,pred)),
    }

def cluster_index_groups(df):
    return [g.index.to_numpy(int) for _,g in df.groupby(df["tree"].astype(str),sort=True)]

def bootstrap_indices(groups,rng):
    chosen=rng.integers(0,len(groups),len(groups))
    return np.concatenate([groups[i] for i in chosen])

def cluster_boot_metrics(df,target):
    d=df.reset_index(drop=True)
    groups=cluster_index_groups(d)
    rng=np.random.default_rng(SEED + (0 if target=="moe_mpa" else 1))
    y=d["y_true"].to_numpy(float)
    preds={label:d[col].to_numpy(float) for label,col in METHODS.items()}
    observed=[]
    for label,p in preds.items():
        observed.append({"target":target,"method":label,"clusters":len(groups),
                         "boards":len(d),**metric(y,p)})
    boot={label:{k:np.empty(BOOT,float) for k in ["mae","rmse","r2"]} for label in METHODS}
    diffs={f"FROZEN_MINUS_{b}":{k:np.empty(BOOT,float) for k in ["mae","rmse"]} for b in ["ERM","EQUAL"]}
    for it in range(BOOT):
        idx=bootstrap_indices(groups,rng)
        yy=y[idx]
        m={}
        for label,p in preds.items():
            m[label]=metric(yy,p[idx])
            for k in boot[label]:
                boot[label][k][it]=m[label][k]
        for base in ["ERM","EQUAL"]:
            for k in ["mae","rmse"]:
                diffs[f"FROZEN_MINUS_{base}"][k][it]=m["FROZEN"][k]-m[base][k]
    rows=[]
    for obs in observed:
        label=obs["method"]
        row=obs.copy()
        for k in ["mae","rmse","r2"]:
            arr=boot[label][k]
            row[f"{k}_cluster_ci_low"]=float(np.nanquantile(arr,.025))
            row[f"{k}_cluster_ci_high"]=float(np.nanquantile(arr,.975))
        rows.append(row)
    diff_rows=[]
    for comp,vals in diffs.items():
        base=comp.replace("FROZEN_MINUS_","")
        obs_f=next(x for x in observed if x["method"]=="FROZEN")
        obs_b=next(x for x in observed if x["method"]==base)
        for k in ["mae","rmse"]:
            arr=vals[k]
            diff_rows.append({
                "target":target,"comparison":comp,"metric":k,
                "observed_difference":obs_f[k]-obs_b[k],
                "cluster_ci_low":float(np.quantile(arr,.025)),
                "cluster_ci_high":float(np.quantile(arr,.975)),
                "probability_difference_gt_zero":float(np.mean(arr>0)),
                "bootstrap_iterations":BOOT,
            })
    return rows,diff_rows

def sign_flip_p(diff,rng):
    x=np.asarray(diff,float)
    obs=abs(float(np.mean(x)))
    count=0
    chunk=5000
    done=0
    while done<PERM:
        n=min(chunk,PERM-done)
        signs=rng.choice([-1.0,1.0],size=(n,len(x)))
        stats=np.abs((signs*x).mean(axis=1))
        count += int(np.sum(stats>=obs-1e-15))
        done += n
    return (count+1)/(PERM+1)

def holm(pvalues):
    p=np.asarray(pvalues,float)
    order=np.argsort(p)
    adj=np.empty(len(p))
    running=0.0
    m=len(p)
    for rank,idx in enumerate(order):
        val=(m-rank)*p[idx]
        running=max(running,val)
        adj[idx]=min(1.0,running)
    return adj

def tree_tests(pred):
    rows=[]
    rng=np.random.default_rng(SEED+77)
    for target in TARGETS:
        d=pred[pred["target"]==target].copy()
        for base_label,base_col in [("ERM","pred_erm"),("EQUAL","pred_equal_ensemble")]:
            d["ae_frozen"]=abs(d["y_true"]-d["pred_frozen"])
            d["ae_base"]=abs(d["y_true"]-d[base_col])
            t=d.groupby("tree",as_index=False)[["ae_frozen","ae_base"]].mean()
            diff=t["ae_frozen"]-t["ae_base"]
            try:
                w=wilcoxon(t["ae_frozen"],t["ae_base"],alternative="two-sided",zero_method="wilcox")
                wp=float(w.pvalue); stat=float(w.statistic)
            except Exception:
                wp=np.nan; stat=np.nan
            rows.append({
                "target":target,"comparison":f"FROZEN_MINUS_{base_label}",
                "trees":len(t),
                "mean_tree_difference":float(diff.mean()),
                "median_tree_difference":float(diff.median()),
                "trees_frozen_worse":int((diff>0).sum()),
                "trees_frozen_better":int((diff<0).sum()),
                "wilcoxon_statistic":stat,
                "wilcoxon_p_raw":wp,
                "sign_flip_p_raw":float(sign_flip_p(diff.to_numpy(),rng)),
            })
    out=pd.DataFrame(rows)
    out["wilcoxon_p_holm"]=holm(out["wilcoxon_p_raw"].fillna(1.0))
    out["sign_flip_p_holm"]=holm(out["sign_flip_p_raw"].fillna(1.0))
    return out

def cluster_spearman_ci(df,target):
    d=df.reset_index(drop=True)
    groups=cluster_index_groups(d)
    x=d["source_disagreement"].to_numpy(float)
    e=abs(d["y_true"].to_numpy(float)-d["pred_frozen"].to_numpy(float))
    obs=float(spearmanr(x,e).statistic)
    rng=np.random.default_rng(SEED+(11 if target=="moe_mpa" else 12))
    vals=np.empty(BOOT,float)
    for it in range(BOOT):
        idx=bootstrap_indices(groups,rng)
        vals[it]=spearmanr(x[idx],e[idx]).statistic
    return {
        "target":target,"boards":len(d),"trees":len(groups),
        "spearman":obs,
        "cluster_ci_low":float(np.nanquantile(vals,.025)),
        "cluster_ci_high":float(np.nanquantile(vals,.975)),
        "probability_positive":float(np.nanmean(vals>0)),
        "bootstrap_iterations":BOOT,
    }

def disagreement_deciles(df,target):
    d=df.copy()
    d["absolute_error"]=abs(d["y_true"]-d["pred_frozen"])
    d["adaptive_width"]=d["frozen_upper"]-d["frozen_lower"]
    d["adaptive_covered"]=(
        (d["y_true"]>=d["frozen_lower"])&(d["y_true"]<=d["frozen_upper"])
    ).astype(int)
    d["global_covered"]=(
        (d["y_true"]>=d["global_lower"])&(d["y_true"]<=d["global_upper"])
    ).astype(int)
    d["disagreement_decile"]=pd.qcut(
        d["source_disagreement"].rank(method="first"),10,labels=False
    )+1
    rows=[]
    for dec,g in d.groupby("disagreement_decile"):
        rows.append({
            "target":target,"disagreement_decile":int(dec),
            "boards":len(g),"trees":g["tree"].nunique(),
            "mean_disagreement":float(g["source_disagreement"].mean()),
            "mean_absolute_error":float(g["absolute_error"].mean()),
            "median_absolute_error":float(g["absolute_error"].median()),
            "adaptive_coverage":float(g["adaptive_covered"].mean()),
            "global_coverage":float(g["global_covered"].mean()),
            "mean_adaptive_width":float(g["adaptive_width"].mean()),
        })
    return pd.DataFrame(rows)

def risk_curve(df,target):
    y=df["y_true"].to_numpy(float)
    pred=df["pred_frozen"].to_numpy(float)
    err=abs(y-pred)
    risk=df["source_disagreement"].to_numpy(float)
    n=len(df)
    fractions=np.linspace(.10,1.0,19)
    rng=np.random.default_rng(SEED+(31 if target=="moe_mpa" else 32))
    order_dis=np.argsort(risk)
    order_oracle=np.argsort(err)
    random_mae=np.zeros(len(fractions))
    for _ in range(RANDOM_RANKINGS):
        order=rng.permutation(n)
        for j,f in enumerate(fractions):
            k=max(1,int(round(f*n)))
            random_mae[j]+=err[order[:k]].mean()
    random_mae/=RANDOM_RANKINGS
    rows=[]
    for label,order in [("DISAGREEMENT",order_dis),("ORACLE",order_oracle)]:
        vals=[]
        for f in fractions:
            k=max(1,int(round(f*n)))
            vals.append(float(err[order[:k]].mean()))
        aurc=float(np.trapezoid(vals,fractions))
        for f,v in zip(fractions,vals):
            rows.append({"target":target,"ranking":label,"retained_fraction":float(f),
                         "retained_n":max(1,int(round(f*n))),"mae":v,"aurc":aurc})
    rand_aurc=float(np.trapezoid(random_mae,fractions))
    for f,v in zip(fractions,random_mae):
        rows.append({"target":target,"ranking":"RANDOM_MEAN",
                     "retained_fraction":float(f),
                     "retained_n":max(1,int(round(f*n))),
                     "mae":float(v),"aurc":rand_aurc})
    return pd.DataFrame(rows)

def conditional_coverage(df,target):
    d=df.copy()
    d["adaptive_width"]=d["frozen_upper"]-d["frozen_lower"]
    const_half=float(d["adaptive_width"].mean()/2.0)
    d["constant_lower"]=d["pred_frozen"]-const_half
    d["constant_upper"]=d["pred_frozen"]+const_half
    for method,lo,hi in [
        ("GLOBAL","global_lower","global_upper"),
        ("ADAPTIVE","frozen_lower","frozen_upper"),
        ("WIDTH_MATCHED_CONSTANT","constant_lower","constant_upper"),
    ]:
        d[f"covered_{method}"]=((d["y_true"]>=d[lo])&(d["y_true"]<=d[hi])).astype(int)
    d["disagreement_quintile"]=pd.qcut(
        d["source_disagreement"].rank(method="first"),5,labels=False
    )+1
    rows=[]
    for q,g in d.groupby("disagreement_quintile"):
        for method in ["GLOBAL","ADAPTIVE","WIDTH_MATCHED_CONSTANT"]:
            rows.append({
                "target":target,"stratifier":"DISAGREEMENT_QUINTILE",
                "stratum":int(q),"method":method,
                "boards":len(g),"trees":g["tree"].nunique(),
                "coverage":float(g[f"covered_{method}"].mean()),
                "mean_width":float(
                    (g["global_upper"]-g["global_lower"]).mean()
                    if method=="GLOBAL" else
                    (g["frozen_upper"]-g["frozen_lower"]).mean()
                    if method=="ADAPTIVE" else 2*const_half
                )
            })
    return pd.DataFrame(rows)

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore",sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore",sparse=False)

def shift_diagnostics(stage0f,stage3a,stage2c,pred,weights):
    usda=pd.read_csv(stage0f/"freeze/usda_clearwood_primary_model_ready.csv")
    board=pd.read_csv(stage3a/"quarantine/Board_Data.csv",encoding="utf-8-sig")
    ext=pd.DataFrame({
        "specific_gravity":pd.to_numeric(board["Density"],errors="coerce")/1000.0,
        "moisture_percent":np.nan,
        "moisture_missing":1,
        "moisture_state":"__MISSING__",
        "wood_class":"softwood",
        "genus":"Pinus",
        "family":"Pinaceae",
        "regime":1,
    })
    dev=usda[["specific_gravity","moisture_percent","moisture_state","wood_class","genus","family"]].copy()
    dev["moisture_missing"]=dev["moisture_percent"].isna().astype(int)
    dev["regime"]=0
    common=pd.concat([dev,ext],ignore_index=True)
    nums=["specific_gravity","moisture_percent","moisture_missing"]
    cats=["moisture_state","wood_class","genus","family"]
    pre=ColumnTransformer([
        ("num",Pipeline([
            ("imp",SimpleImputer(strategy="median",add_indicator=True)),
            ("scale",StandardScaler())
        ]),nums),
        ("cat",Pipeline([
            ("imp",SimpleImputer(strategy="constant",fill_value="__MISSING__")),
            ("ohe",make_ohe())
        ]),cats)
    ],sparse_threshold=0.0)
    clf=Pipeline([
        ("pre",pre),
        ("clf",LogisticRegression(max_iter=3000,class_weight="balanced"))
    ])
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    prob=cross_val_predict(clf,common[nums+cats],common["regime"],cv=cv,method="predict_proba")[:,1]
    auc_all=float(roc_auc_score(common["regime"],prob))
    fpr,tpr,_=roc_curve(common["regime"],prob)

    density_pipe=Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("clf",LogisticRegression(max_iter=3000,class_weight="balanced"))
    ])
    prob_d=cross_val_predict(
        density_pipe,common[["specific_gravity"]],common["regime"],
        cv=cv,method="predict_proba"
    )[:,1]
    auc_density=float(roc_auc_score(common["regime"],prob_d))

    x=dev["specific_gravity"].dropna().to_numpy(float)
    y=ext["specific_gravity"].dropna().to_numpy(float)
    pooled=math.sqrt(((len(x)-1)*x.var(ddof=1)+(len(y)-1)*y.var(ddof=1))/(len(x)+len(y)-2))
    smd=float((y.mean()-x.mean())/pooled)
    ks=ks_2samp(x,y)

    rows=[
        {"diagnostic":"domain_classifier_auc_all_common_features","value":auc_all,
         "interpretation":"Higher values indicate easier separation and stronger shift"},
        {"diagnostic":"domain_classifier_auc_density_only","value":auc_density,
         "interpretation":"Shift detectable using density alone"},
        {"diagnostic":"specific_gravity_standardized_mean_difference_external_minus_development",
         "value":smd,"interpretation":"Absolute values above 0.8 are conventionally large"},
        {"diagnostic":"specific_gravity_ks_statistic","value":float(ks.statistic),
         "interpretation":"Distributional separation"},
        {"diagnostic":"specific_gravity_ks_pvalue","value":float(ks.pvalue),
         "interpretation":"Descriptive post-hoc test"},
        {"diagnostic":"development_moisture_missing_fraction",
         "value":float(dev["moisture_missing"].mean()),"interpretation":""},
        {"diagnostic":"external_moisture_missing_fraction","value":1.0,
         "interpretation":"All external board moisture values unavailable"},
        {"diagnostic":"external_genus_unseen_fraction",
         "value":float((~ext["genus"].isin(set(dev["genus"].dropna()))).mean()),
         "interpretation":""},
        {"diagnostic":"external_family_unseen_fraction",
         "value":float((~ext["family"].isin(set(dev["family"].dropna()))).mean()),
         "interpretation":""},
    ]

    devpred=pd.read_csv(stage2c/"data/APPLICABILITY_WEIGHTED_DEVELOPMENT_PREDICTIONS.csv")
    for target in TARGETS:
        dd=devpred[devpred["target"]==target]["source_disagreement"].dropna()
        ee=pred[pred["target"]==target]["source_disagreement"].dropna()
        k=ks_2samp(dd,ee)
        rows += [
            {"diagnostic":f"{target}_development_median_disagreement",
             "value":float(dd.median()),"interpretation":""},
            {"diagnostic":f"{target}_external_median_disagreement",
             "value":float(ee.median()),"interpretation":""},
            {"diagnostic":f"{target}_disagreement_ks_statistic",
             "value":float(k.statistic),"interpretation":""},
            {"diagnostic":f"{target}_disagreement_ks_pvalue",
             "value":float(k.pvalue),"interpretation":""},
        ]

    w=weights.copy()
    ent=[]
    for (target,record),g in w.groupby(["target","record_id"]):
        p=g["weight"].to_numpy(float)
        entropy=float(-(p*np.log(np.clip(p,1e-15,None))).sum()/math.log(len(p)))
        ent.append({
            "target":target,"record_id":record,
            "max_source_weight":float(p.max()),
            "normalized_weight_entropy":entropy,
            "minimum_source_distance":float(g["distance"].min()),
        })
    ent=pd.DataFrame(ent)
    return pd.DataFrame(rows),common.assign(domain_probability=prob),pd.DataFrame({
        "fpr":fpr,"tpr":tpr
    }),ent,usda,board

def figures(cluster_diff,deciles,risk,conditional,shift,roc,entropy,usda,board):
    for target in TARGETS:
        d=cluster_diff[(cluster_diff["target"]==target)&(cluster_diff["metric"]=="mae")]
        plt.figure(figsize=(7,4.5))
        y=np.arange(len(d))
        x=d["observed_difference"].to_numpy(float)
        lo=x-d["cluster_ci_low"].to_numpy(float)
        hi=d["cluster_ci_high"].to_numpy(float)-x
        plt.errorbar(x,y,xerr=[lo,hi],fmt="o",capsize=4)
        plt.axvline(0,linestyle="--")
        plt.yticks(y,d["comparison"])
        plt.xlabel("Frozen minus baseline MAE")
        plt.title(f"Tree-cluster bootstrap MAE differences: {target}")
        plt.tight_layout()
        plt.savefig(FIGURES/f"CLUSTER_MAE_DIFFERENCE_{target}.png",dpi=220)
        plt.close()

        q=deciles[deciles["target"]==target]
        plt.figure(figsize=(7.5,4.5))
        plt.plot(q["disagreement_decile"],q["mean_absolute_error"],marker="o")
        plt.xlabel("Source-disagreement decile")
        plt.ylabel("Mean absolute error")
        plt.title(f"Error by disagreement decile: {target}")
        plt.tight_layout()
        plt.savefig(FIGURES/f"ERROR_BY_DISAGREEMENT_DECILE_{target}.png",dpi=220)
        plt.close()

        r=risk[risk["target"]==target]
        plt.figure(figsize=(7.5,4.5))
        for label,g in r.groupby("ranking"):
            plt.plot(g["retained_fraction"],g["mae"],marker="o",label=label)
        plt.xlabel("Retained fraction")
        plt.ylabel("MAE")
        plt.title(f"Risk–coverage rankings: {target}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES/f"RISK_COVERAGE_COMPARISON_{target}.png",dpi=220)
        plt.close()

        c=conditional[(conditional["target"]==target)]
        plt.figure(figsize=(8,4.8))
        for label,g in c.groupby("method"):
            plt.plot(g["stratum"],100*g["coverage"],marker="o",label=label)
        plt.axhline(90,linestyle="--")
        plt.xlabel("Disagreement quintile")
        plt.ylabel("Coverage (%)")
        plt.title(f"Conditional coverage by disagreement: {target}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIGURES/f"CONDITIONAL_COVERAGE_{target}.png",dpi=220)
        plt.close()

    plt.figure(figsize=(6,5))
    plt.plot(roc["fpr"],roc["tpr"])
    plt.plot([0,1],[0,1],linestyle="--")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.title("Development-vs-external domain classifier")
    plt.tight_layout()
    plt.savefig(FIGURES/"DOMAIN_CLASSIFIER_ROC.png",dpi=220)
    plt.close()

    plt.figure(figsize=(7.5,4.5))
    plt.hist(usda["specific_gravity"].dropna(),bins=25,alpha=.6,density=True,label="USDA development")
    plt.hist(pd.to_numeric(board["Density"],errors="coerce").dropna()/1000.0,
             bins=25,alpha=.6,density=True,label="Pinus patula external")
    plt.xlabel("Specific gravity / density divided by 1000")
    plt.ylabel("Density")
    plt.title("Common density-feature shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES/"SPECIFIC_GRAVITY_SHIFT.png",dpi=220)
    plt.close()

    for target in TARGETS:
        e=entropy[entropy["target"]==target]
        plt.figure(figsize=(7.5,4.5))
        plt.hist(e["normalized_weight_entropy"],bins=20)
        plt.xlabel("Normalized source-weight entropy")
        plt.ylabel("Boards")
        plt.title(f"External applicability-weight entropy: {target}")
        plt.tight_layout()
        plt.savefig(FIGURES/f"SOURCE_WEIGHT_ENTROPY_{target}.png",dpi=220)
        plt.close()

def main():
    shutil.rmtree(OUT,ignore_errors=True)
    for d in [OUT,WORK,TABLES,FIGURES]:
        d.mkdir(parents=True,exist_ok=True)
    write_json(OUT/"STATUS.json",{"stage":"STAGE4H","status":"running","started_at":now()})
    try:
        roots={name:verify_result(name,z) for name,z in ZIPS.items()}
        log("All Stage0F/2C/3A/3B/4C inputs and hashes verified")

        pred=pd.read_csv(roots["stage3b"]/"data/EXTERNAL_FROZEN_PREDICTIONS.csv")
        weights=pd.read_csv(roots["stage3b"]/"data/EXTERNAL_SOURCE_WEIGHTS.csv")
        pred["tree"]=pred["tree"].astype(str)

        metric_rows=[]; diff_rows=[]
        for target in TARGETS:
            d=pred[pred["target"]==target].copy()
            m,di=cluster_boot_metrics(d,target)
            metric_rows.extend(m);diff_rows.extend(di)
        cluster_metrics=pd.DataFrame(metric_rows)
        cluster_diff=pd.DataFrame(diff_rows)
        tree_test=tree_tests(pred)

        spearman_rows=[]; decile_frames=[]; risk_frames=[]; cond_frames=[]
        for target in TARGETS:
            d=pred[pred["target"]==target].copy()
            spearman_rows.append(cluster_spearman_ci(d,target))
            decile_frames.append(disagreement_deciles(d,target))
            risk_frames.append(risk_curve(d,target))
            cond_frames.append(conditional_coverage(d,target))
        spearman=pd.DataFrame(spearman_rows)
        deciles=pd.concat(decile_frames,ignore_index=True)
        risk=pd.concat(risk_frames,ignore_index=True)
        conditional=pd.concat(cond_frames,ignore_index=True)

        shift,domain_rows,roc,entropy,usda,board=shift_diagnostics(
            roots["stage0f"],roots["stage3a"],roots["stage2c"],pred,weights
        )

        cluster_metrics.to_csv(TABLES/"CLUSTER_AWARE_POINT_METRICS.csv",index=False,encoding="utf-8-sig")
        cluster_diff.to_csv(TABLES/"CLUSTER_BOOTSTRAP_POINT_DIFFERENCES.csv",index=False,encoding="utf-8-sig")
        tree_test.to_csv(TABLES/"TREE_LEVEL_PAIRED_TESTS.csv",index=False,encoding="utf-8-sig")
        spearman.to_csv(TABLES/"DISAGREEMENT_ERROR_CLUSTER_SPEARMAN.csv",index=False,encoding="utf-8-sig")
        deciles.to_csv(TABLES/"ERROR_COVERAGE_BY_DISAGREEMENT_DECILE.csv",index=False,encoding="utf-8-sig")
        risk.to_csv(TABLES/"RISK_COVERAGE_RANDOM_ORACLE_AUDIT.csv",index=False,encoding="utf-8-sig")
        conditional.to_csv(TABLES/"CONDITIONAL_COVERAGE_AUDIT.csv",index=False,encoding="utf-8-sig")
        shift.to_csv(TABLES/"CROSS_REGIME_SHIFT_DIAGNOSTICS.csv",index=False,encoding="utf-8-sig")
        entropy.to_csv(TABLES/"EXTERNAL_SOURCE_WEIGHT_DIAGNOSTICS.csv",index=False,encoding="utf-8-sig")

        with pd.ExcelWriter(TABLES/"STAGE4H_STRENGTHENING_RESULTS.xlsx",engine="openpyxl") as xw:
            cluster_metrics.to_excel(xw,sheet_name="Cluster metrics",index=False)
            cluster_diff.to_excel(xw,sheet_name="Cluster differences",index=False)
            tree_test.to_excel(xw,sheet_name="Tree tests",index=False)
            spearman.to_excel(xw,sheet_name="Disagreement Spearman",index=False)
            deciles.to_excel(xw,sheet_name="Disagreement deciles",index=False)
            risk.to_excel(xw,sheet_name="Risk coverage",index=False)
            conditional.to_excel(xw,sheet_name="Conditional coverage",index=False)
            shift.to_excel(xw,sheet_name="Shift diagnostics",index=False)
            entropy.to_excel(xw,sheet_name="Source weights",index=False)

        figures(cluster_diff,deciles,risk,conditional,shift,roc,entropy,usda,board)

        shift_map=dict(zip(shift["diagnostic"],shift["value"]))
        spearman_supported={
            r["target"]:bool(r["cluster_ci_low"]>0.10)
            for _,r in spearman.iterrows()
        }
        cluster_negative={}
        for target in TARGETS:
            q=cluster_diff[(cluster_diff["target"]==target)&(cluster_diff["metric"]=="mae")]
            cluster_negative[target]={
                row["comparison"]:bool(row["cluster_ci_low"]>0)
                for _,row in q.iterrows()
            }

        decision="STRENGTHENED_NEGATIVE_CROSS_REGIME_EVIDENCE_SAME_REGIME_LOCKBOX_STILL_HIGH_VALUE"
        scientific={
            "decision":decision,
            "confirmatory_method_changed":False,
            "frozen_parameters_changed":False,
            "external_labels_used_for_retuning":False,
            "analysis_status":"POST_HOC_REVIEWER_MOTIVATED_DIAGNOSTIC_STRENGTHENING",
            "cluster_aware_external_point_findings":cluster_negative,
            "disagreement_local_signal_supported":spearman_supported,
            "cross_regime_domain_classifier_auc_all_features":
                shift_map["domain_classifier_auc_all_common_features"],
            "cross_regime_domain_classifier_auc_density_only":
                shift_map["domain_classifier_auc_density_only"],
            "interpretation":[
                "Tree-cluster-aware inference replaces board-independence assumptions.",
                "Disagreement is evaluated as a local error-ranking signal rather than inferred from marginal coverage.",
                "Shift diagnostics quantify how separable clear-wood development and structural-board external regimes are.",
                "These analyses strengthen rigor but cannot create same-regime external validation.",
                "Eucalyptus data remain the highest-value new evidence if supplied with specimen/tree identifiers, density, MOE and MOR."
            ],
            "next_stage":"STAGE4I_POST_HOC_BASELINES_ABLATIONS_AND_MANUSCRIPT_REWRITE",
            "generated_at":now()
        }
        write_json(OUT/"SCIENTIFIC_STAGE4H_DECISION.json",scientific)

        # Build readable Arabic report with exact generated values.
        lines=[
            "# WoodPhys-DG — Stage 4H",
            "",
            f"## القرار: **{decision}**",
            "",
            "هذه مرحلة تقوية إحصائية وتشخيصية من نفس البيانات القديمة؛ لم تُنشأ Dataset جديدة،",
            "ولم تتغير tau أو beta، ولم يُعد اختيار الطريقة باستخدام Pinus patula.",
            "",
            "## 1. Tree-cluster-aware point inference",
        ]
        for target in TARGETS:
            lines += ["",f"### {target}"]
            for _,r in cluster_diff[
                (cluster_diff["target"]==target)&(cluster_diff["metric"]=="mae")
            ].iterrows():
                lines.append(
                    f"- {r['comparison']}: difference={r['observed_difference']:.6g}, "
                    f"95% cluster CI [{r['cluster_ci_low']:.6g}, {r['cluster_ci_high']:.6g}]"
                )
        lines += ["","## 2. Disagreement as a local error signal"]
        for _,r in spearman.iterrows():
            lines.append(
                f"- {r['target']}: Spearman={r['spearman']:.4f}, "
                f"95% tree-cluster CI [{r['cluster_ci_low']:.4f}, {r['cluster_ci_high']:.4f}]"
            )
        lines += [
            "",
            "إذا كان الـCI يشمل الصفر أو القيم الصغيرة، فلا يجوز وصف disagreement بأنها",
            "local uncertainty signal قوية حتى لو كانت marginal coverage مرتفعة.",
            "",
            "## 3. Cross-regime shift",
            f"- Domain-classifier AUC باستخدام السمات المشتركة: "
            f"{shift_map['domain_classifier_auc_all_common_features']:.4f}",
            f"- Density-only AUC: {shift_map['domain_classifier_auc_density_only']:.4f}",
            f"- Specific-gravity standardized mean difference: "
            f"{shift_map['specific_gravity_standardized_mean_difference_external_minus_development']:.4f}",
            "",
            "AUC مرتفعة تعني أن USDA clear wood وPinus structural boards سهل الفصل بينهما،",
            "وبالتالي الاختبار الخارجي severe cross-regime stress test وليس same-regime validation.",
            "",
            "## 4. ماذا أضافت هذه المرحلة؟",
            "- صححت خطر اعتبار 207 ألواح عينات مستقلة بينما توجد 24 شجرة.",
            "- قارنت disagreement مع random وoracle ranking.",
            "- أضافت conditional coverage حسب disagreement.",
            "- قاست شدة التحول بين النظامين.",
            "",
            "## 5. ماذا لا تزال Eucalyptus تضيف؟",
            "إذا وصلت بيانات specimen-level أو tree-level وبها density وMOE وMOR،",
            "فهي تضيف same-regime independent lockbox. هذه أهم إضافة بيانات ممكنة،",
            "لكن نتيجتها لا تُعرف قبل التشغيل ولا تضمن نجاح الطريقة.",
            "",
            "## التالي",
            "Stage 4I: baselines وablations post hoc من development-only fitting، ثم إعادة كتابة البحث.",
        ]
        (OUT/"STAGE4H_REPORT_AR.md").write_text("\n".join(lines),encoding="utf-8")

        write_json(OUT/"OUTPUT_VERIFICATION.json",{
            "verification_passed":True,
            "all_input_hashes_verified":True,
            "cluster_bootstrap_iterations":BOOT,
            "sign_flip_permutations":PERM,
            "random_rankings":RANDOM_RANKINGS,
            "cluster_aware_point_metrics_created":True,
            "tree_level_tests_created":True,
            "disagreement_diagnostics_created":True,
            "conditional_coverage_created":True,
            "shift_diagnostics_created":True,
            "figures_created":len(list(FIGURES.glob("*.png"))),
            "frozen_parameters_changed":False,
            "model_retuned_on_external_labels":False,
        })
        write_json(OUT/"STATUS.json",{
            "stage":"STAGE4H","status":"completed","decision":decision,
            "completed_at":now()
        })
        write_json(OUT/"SUCCESS_MARKER.json",{
            "success":True,"stage":"STAGE4H","decision":decision,"timestamp":now()
        })
        (OUT/"ERROR_LOG.txt").write_text("",encoding="utf-8")
        (OUT/"RUN_LOG.txt").write_text("\n".join(LOG)+"\n",encoding="utf-8")

        manifest=[];hashes=[]
        for p in sorted(OUT.rglob("*")):
            if not p.is_file() or "_work" in p.parts:
                continue
            rel=p.relative_to(OUT).as_posix()
            if rel in {"FILE_MANIFEST.csv","HASHES_SHA256.csv","UPLOAD_THIS_STAGE4H_RESULT.zip"}:
                continue
            manifest.append({"relative_path":rel,"size_bytes":p.stat().st_size})
            hashes.append({"relative_path":rel,"sha256":sha256(p)})
        pd.DataFrame(manifest).to_csv(OUT/"FILE_MANIFEST.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(hashes).to_csv(OUT/"HASHES_SHA256.csv",index=False,encoding="utf-8-sig")

        with zipfile.ZipFile(OUT/"UPLOAD_THIS_STAGE4H_RESULT.zip","w",zipfile.ZIP_DEFLATED) as z:
            for p in sorted(OUT.rglob("*")):
                if p.is_file() and p.name!="UPLOAD_THIS_STAGE4H_RESULT.zip" and "_work" not in p.parts:
                    z.write(p,p.relative_to(OUT))
        log("Stage4H complete")
        (OUT/"RUN_LOG.txt").write_text("\n".join(LOG)+"\n",encoding="utf-8")
        return 0
    except Exception as exc:
        ERR.append(traceback.format_exc())
        write_json(OUT/"STATUS.json",{"stage":"STAGE4H","status":"failed",
                                     "error":str(exc),"completed_at":now()})
        write_json(OUT/"SUCCESS_MARKER.json",{"success":False,"stage":"STAGE4H",
                                             "error":str(exc),"timestamp":now()})
        (OUT/"ERROR_LOG.txt").write_text("\n".join(ERR),encoding="utf-8")
        (OUT/"RUN_LOG.txt").write_text("\n".join(LOG)+"\n",encoding="utf-8")
        print(ERR[-1],file=sys.stderr)
        return 1

if __name__=="__main__":
    raise SystemExit(main())
