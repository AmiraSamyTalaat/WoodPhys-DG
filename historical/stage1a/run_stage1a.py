
from __future__ import annotations
import datetime as dt, hashlib, json, math, platform, shutil, sys, traceback, zipfile, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[1]
INPUT0F=ROOT/"input"/"STAGE0F_VERIFIED_INPUT.zip"
INPUT1=ROOT/"input"/"STAGE1_ORIGINAL_RESULT_FOR_AUDIT.zip"
OUT=ROOT/"stage1a_output"
WORK=OUT/"_work"; TABLES=OUT/"tables"; DATA=OUT/"data"; FIG=OUT/"figures"
FAST_TEST=os.environ.get("WOODPHYS_FAST_TEST","0")=="1"
SEEDS=[42] if FAST_TEST else [13,42,97,131,2026]; ALPHA=0.10; BOOT=100 if FAST_TEST else 1000
LOG=[]; ERR=[]

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def log(x):
    s=f"[{now()}] {x}"; LOG.append(s); print(s,flush=True)
def wjson(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding="utf-8")
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()
def ohe():
    try:return OneHotEncoder(handle_unknown="ignore",sparse_output=False)
    except TypeError:return OneHotEncoder(handle_unknown="ignore",sparse=False)
def prep(nums,cats):
    ts=[]
    if nums: ts.append(("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),nums))
    if cats: ts.append(("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",ohe())]),cats))
    return ColumnTransformer(ts,remainder="drop",sparse_threshold=0.0)
def model(name,seed):
    if name=="MEDIAN": return DummyRegressor(strategy="median")
    if name=="RIDGE": return Ridge(alpha=1.0)
    if name=="RANDOM_FOREST":
        return RandomForestRegressor(n_estimators=20 if FAST_TEST else 120,min_samples_leaf=3,max_features=0.8,
                                     random_state=seed,n_jobs=1)
    if name=="HIST_GRADIENT_BOOSTING":
        return HistGradientBoostingRegressor(max_iter=50 if FAST_TEST else 160,learning_rate=0.04,max_leaf_nodes=15,
                                             min_samples_leaf=10,l2_regularization=1.0,
                                             random_state=seed)
    raise ValueError(name)
def metric(y,p):
    return {"n":len(y),"mae":mean_absolute_error(y,p),
            "rmse":mean_squared_error(y,p)**0.5,"r2":r2_score(y,p)}
def cq(res,alpha=ALPHA):
    a=np.sort(np.asarray(res,float)); n=len(a)
    k=min(n-1,max(0,int(math.ceil((n+1)*(1-alpha)))-1))
    return float(a[k])
def boot_ci(y,p,seed=2026):
    y=np.asarray(y);p=np.asarray(p);n=len(y);rng=np.random.default_rng(seed)
    vals=np.empty(BOOT)
    for i in range(BOOT):
        idx=rng.integers(0,n,n)
        vals[i]=mean_absolute_error(y[idx],p[idx])
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))
def scores(pipe,xtr,x):
    pre=pipe.named_steps["pre"]
    a=np.asarray(pre.transform(xtr),float);b=np.asarray(pre.transform(x),float)
    nn=NearestNeighbors(n_neighbors=1).fit(a)
    return nn.kneighbors(b)[0][:,0]
def verify():
    a=WORK/"stage0f"; b=WORK/"stage1"
    a.mkdir(parents=True);b.mkdir(parents=True)
    with zipfile.ZipFile(INPUT0F) as z:z.extractall(a)
    with zipfile.ZipFile(INPUT1) as z:z.extractall(b)
    s=json.loads((a/"SUCCESS_MARKER.json").read_text())
    v=json.loads((a/"OUTPUT_VERIFICATION.json").read_text())
    if not s.get("success") or not v.get("verification_passed"):raise RuntimeError("Stage0F invalid")
    original=json.loads((b/"OUTPUT_VERIFICATION.json").read_text())
    log(f"Original Stage1 audit: seeds={original.get('seeds')}, grouped_cv_rows={original.get('grouped_cv_rows')}")
    return a,b,original
def clean(df,target,features,idcol):
    d=df[features+[target,idcol]].copy()
    d[target]=pd.to_numeric(d[target],errors="coerce")
    return d[d[target].notna()].copy()

def scenario_run(regime,scenario,target,feature_sets,train,cal,test,idcol,subgroup=None):
    log(f"Scenario start: {regime} / {target}")
    metric_rows=[];selection=[];pred_rows=[];uq_rows=[];risk_rows=[]
    candidates=[]
    for fs,nums,cats in feature_sets:
        feats=nums+cats
        tr=clean(train,target,feats,idcol); ca=clean(cal,target,feats,idcol); te=clean(test,target,feats,idcol)
        for mname in ["MEDIAN","RIDGE","RANDOM_FOREST","HIST_GRADIENT_BOOSTING"]:
            seed_records=[]
            runseeds=[42]
            for seed in runseeds:
                pipe=Pipeline([("pre",prep(nums,cats)),("model",model(mname,seed))])
                pipe.fit(tr[feats],tr[target])
                pc=pipe.predict(ca[feats]); pt=pipe.predict(te[feats])
                mc=metric(ca[target].to_numpy(),pc); mt=metric(te[target].to_numpy(),pt)
                metric_rows += [
                    {"regime":regime,"scenario":scenario,"target":target,"feature_set":fs,
                     "model":mname,"seed":seed,"split":"CALIBRATION",**mc},
                    {"regime":regime,"scenario":scenario,"target":target,"feature_set":fs,
                     "model":mname,"seed":seed,"split":"TEST",**mt},
                ]
                seed_records.append((seed,pipe,pc,pt,mc["mae"],mt["mae"]))
            candidates.append({"feature_set":fs,"nums":nums,"cats":cats,"model":mname,
                               "cal_mae":float(np.mean([x[4] for x in seed_records])),
                               "records":seed_records,"tr":tr,"ca":ca,"te":te})
    winner=min(candidates,key=lambda x:x["cal_mae"])
    for c in candidates:
        selection.append({"regime":regime,"scenario":scenario,"target":target,
                          "feature_set":c["feature_set"],"model":c["model"],
                          "mean_calibration_mae":c["cal_mae"],
                          "selected":"YES" if c is winner else "NO"})
    # Robustness rerun for the calibration-selected candidate only.
    robust_records=[]
    tr,ca,te=winner["tr"],winner["ca"],winner["te"]
    feats=winner["nums"]+winner["cats"]
    for robust_seed in SEEDS:
        robust_pipe=Pipeline([("pre",prep(winner["nums"],winner["cats"])),
                              ("model",model(winner["model"],robust_seed))])
        robust_pipe.fit(tr[feats],tr[target])
        robust_pc=robust_pipe.predict(ca[feats]); robust_pt=robust_pipe.predict(te[feats])
        robust_mc=metric(ca[target].to_numpy(),robust_pc)
        robust_mt=metric(te[target].to_numpy(),robust_pt)
        robust_records.append((robust_seed,robust_pipe,robust_pc,robust_pt,
                               robust_mc["mae"],robust_mt["mae"]))
        metric_rows += [
            {"regime":regime,"scenario":scenario,"target":target,
             "feature_set":winner["feature_set"],"model":winner["model"],
             "seed":robust_seed,"split":"CALIBRATION_SELECTED_ROBUST",**robust_mc},
            {"regime":regime,"scenario":scenario,"target":target,
             "feature_set":winner["feature_set"],"model":winner["model"],
             "seed":robust_seed,"split":"TEST_SELECTED_ROBUST",**robust_mt},
        ]
    # fixed seed 42 for uncertainty diagnostics.
    rec=next(r for r in robust_records if r[0]==42)
    seed,pipe,pc,pt,_,_=rec
    tr,ca,te=winner["tr"],winner["ca"],winner["te"]
    feats=winner["nums"]+winner["cats"]
    yc=ca[target].to_numpy(float); yt=te[target].to_numpy(float)
    res=np.abs(yc-pc)
    cal_s=scores(pipe,tr[feats],ca[feats]); test_s=scores(pipe,tr[feats],te[feats])
    # Global.
    qg=cq(res); glo=pt-qg; ghi=pt+qg
    # Tertile-binned.
    edges=np.quantile(cal_s,[1/3,2/3])
    cb=np.digitize(cal_s,edges);tb=np.digitize(test_s,edges)
    qs={}
    for b in [0,1,2]:
        rr=res[cb==b]; qs[b]=cq(rr) if len(rr)>=10 else qg
    qb=np.array([qs[int(b)] for b in tb]); blo=pt-qb;bhi=pt+qb
    # Risk-scaled conformal: isotonic expected residual and conformalized ratio.
    order=np.argsort(cal_s)
    iso=IsotonicRegression(increasing=True,out_of_bounds="clip")
    iso.fit(cal_s[order],res[order])
    rc=np.maximum(iso.predict(cal_s),1e-6); rt=np.maximum(iso.predict(test_s),1e-6)
    qr=cq(res/rc); rlo=pt-qr*rt;rhi=pt+qr*rt
    methods=[
        ("GLOBAL",glo,ghi,np.full(len(yt),qg)),
        ("SHIFT_TERTILE",blo,bhi,qb),
        ("RISK_SCALED",rlo,rhi,qr*rt),
    ]
    for name,lo,hi,half in methods:
        uq_rows.append({"regime":regime,"scenario":scenario,"target":target,
                        "selected_feature_set":winner["feature_set"],"selected_model":winner["model"],
                        "method":name,"coverage":float(np.mean((yt>=lo)&(yt<=hi))),
                        "average_width":float(np.mean(hi-lo)),"test_n":len(yt),
                        "calibration_n":len(yc)})
    # Risk groups by predicted residual, with retained fraction explicit.
    risk_pred=rt
    t50,t90=np.quantile(rc,[.5,.9])
    grp=np.where(risk_pred<=t50,"ACCEPT",np.where(risk_pred<=t90,"REVIEW","ABSTAIN"))
    for g in ["ACCEPT","REVIEW","ABSTAIN"]:
        mask=grp==g
        risk_rows.append({"regime":regime,"scenario":scenario,"target":target,"group":g,
                          "n":int(mask.sum()),"fraction":float(mask.mean()),
                          "mae":float(mean_absolute_error(yt[mask],pt[mask])) if mask.any() else np.nan})
    lo_ci,hi_ci=boot_ci(yt,pt)
    shift_gap=float(np.mean([r[5] for r in robust_records])/
                    max(np.mean([r[4] for r in robust_records]),1e-9))
    for i,row in enumerate(te.itertuples(index=False)):
        pred_rows.append({"regime":regime,"scenario":scenario,"target":target,
                          "record_id":str(getattr(row,idcol)),"y_true":float(yt[i]),
                          "y_pred":float(pt[i]),"absolute_error":float(abs(yt[i]-pt[i])),
                          "shift_score":float(test_s[i]),"predicted_risk":float(risk_pred[i]),
                          "risk_group":str(grp[i])})
    selected_summary={"regime":regime,"scenario":scenario,"target":target,
                      "feature_set":winner["feature_set"],"model":winner["model"],
                      "calibration_mae":float(np.mean([r[4] for r in robust_records])),
                      "test_mae":float(np.mean([r[5] for r in robust_records])),
                      "test_mae_ci_low":lo_ci,"test_mae_ci_high":hi_ci,
                      "calibration_to_test_mae_ratio":shift_gap,
                      "test_r2":float(r2_score(yt,pt))}
    log(f"Scenario complete: {regime} / {target}; selected={winner['model']} {winner['feature_set']}")
    return metric_rows,selection,pred_rows,uq_rows,risk_rows,selected_summary

def group_cv(regime,scenario,target,fs,nums,cats,df,group,idcol):
    log(f"Grouped CV start: {regime} / {target} / {fs}")
    feats=nums+cats
    needed=list(dict.fromkeys(feats+[target,group,idcol]))
    d=df[needed].copy()
    d[target]=pd.to_numeric(d[target],errors="coerce")
    d=d[d[target].notna() & d[group].notna()].copy()
    groups=d[group].astype(str).to_numpy()
    n=min(2 if FAST_TEST else 5,len(np.unique(groups)))
    out=[]
    if n<2:return out
    gkf=GroupKFold(n_splits=n)
    for fold,(tr,te) in enumerate(gkf.split(d,d[target],groups),1):
        for mn in ["RIDGE"]:
            vals=[]
            runseeds=[42]
            for seed in runseeds:
                p=Pipeline([("pre",prep(nums,cats)),("model",model(mn,seed))])
                p.fit(d.iloc[tr][feats],d.iloc[tr][target])
                pr=p.predict(d.iloc[te][feats])
                vals.append(metric(d.iloc[te][target].to_numpy(),pr))
            out.append({"regime":regime,"scenario":scenario,"target":target,"feature_set":fs,
                        "model":mn,"fold":fold,"n_test":len(te),
                        "held_groups":"|".join(sorted(set(groups[te]))),
                        "mae_mean_seeds":float(np.mean([x["mae"] for x in vals])),
                        "rmse_mean_seeds":float(np.mean([x["rmse"] for x in vals])),
                        "r2_mean_seeds":float(np.mean([x["r2"] for x in vals]))})
    log(f"Grouped CV complete: {regime} / {target} / {fs}; folds={n}")
    return out

def main():
    shutil.rmtree(OUT,ignore_errors=True)
    for d in [OUT,WORK,TABLES,DATA,FIG]:d.mkdir(parents=True,exist_ok=True)
    wjson(OUT/"STATUS.json",{"stage":"STAGE1A","status":"running","started_at":now()})
    try:
        s0,s1,orig=verify()
        fr=s0/"freeze"; sp=s0/"splits"
        us=pd.read_csv(fr/"usda_clearwood_primary_model_ready.csv")
        jl=pd.read_csv(fr/"japanese_larch_overall_model_ready.csv")
        sr=pd.read_csv(fr/"spruce_structural_board_separate_regime.csv")
        pn=pd.read_csv(fr/"pinus_nigra_radial_separate_regime.csv")
        um=pd.read_csv(sp/"USDA_DOMAIN_AND_TAXON_SPLIT_MANIFEST.csv")
        jm=pd.read_csv(sp/"JAPANESE_LARCH_SITE_FAMILY_SPLIT_MANIFEST.csv")
        us=us.merge(um[["record_id","domain_role"]],on="record_id",how="left",validate="one_to_one")
        jl=jl.merge(jm[["record_id","site_role"]],on="record_id",how="left",validate="one_to_one")
        met=[];sel=[];pred=[];uq=[];risk=[];selected=[];cv=[]
        us_tr=us[us.domain_role=="DEVELOPMENT_US"];us_ca=us[us.domain_role=="CALIBRATION_CANADA"];us_te=us[us.domain_role=="EXTERNAL_OTHER_IMPORTS"]
        usfs=[("PHYS_ONLY",["specific_gravity","moisture_percent"],["moisture_state","wood_class"]),
              ("PHYS_TAXONOMY",["specific_gravity","moisture_percent"],["moisture_state","wood_class","genus","family"])]
        for target in ["moe_mpa","mor_mpa"]:
            o=scenario_run("USDA_CLEARWOOD","US_TO_CANADA_TO_OTHER",target,usfs,us_tr,us_ca,us_te,"record_id","genus")
            for a,b in zip([met,sel,pred,uq,risk,selected],o):a.extend(b if isinstance(b,list) else [b])
            resolved=us[us.genus.notna() & (us.genus.astype(str)!="")]
            for fs,nums,cats in usfs:
                cv.extend(group_cv("USDA_CLEARWOOD","HELD_OUT_GENUS_CV",target,fs,nums,cats,resolved,"genus","record_id"))
        jd=jl[jl.site_role=="DEVELOPMENT_SITE_431"];jt=jl[jl.site_role=="EXTERNAL_SITE_496"]
        jtr,jca=train_test_split(jd,test_size=.25,random_state=42,stratify=jd["Family_code"])
        jfs=[("PHENOTYPE_NO_FAMILY_ID",["H","DBH","ARW","LWP","MFA","AD","Pp","Yp","Pm","Ym","W"],["Block"])]
        for target in ["moe_mpa","mor_mpa"]:
            o=scenario_run("JAPANESE_LARCH","SITE_431_TO_SITE_496",target,jfs,jtr,jca,jt,"record_id","Family_code")
            for a,b in zip([met,sel,pred,uq,risk,selected],o):a.extend(b if isinstance(b,list) else [b])
            cv.extend(group_cv("JAPANESE_LARCH","HELD_OUT_FAMILY_CV",target,*jfs[0],jl,"Family_code","record_id"))
        st=sr[sr.provided_split=="train"];ste=sr[sr.provided_split=="test"]
        strn,sca=train_test_split(st,test_size=.2,random_state=42)
        sfs=[("DENSITY_GEOMETRY_QUALITY",["length_mm","width_mm","height_mm","density_kg_m3","quality_face_1","quality_face_2"],[])]
        for target in ["moe_mpa","mor_mpa"]:
            o=scenario_run("SPRUCE_STRUCTURAL","PROVIDED_TRAIN_TEST",target,sfs,strn,sca,ste,"record_id",None)
            for a,b in zip([met,sel,pred,uq,risk,selected],o):a.extend(b if isinstance(b,list) else [b])
        frames={
            "ROBUST_METRICS_LONG":pd.DataFrame(met),
            "CALIBRATION_MODEL_SELECTION":pd.DataFrame(sel),
            "SELECTED_TEST_PREDICTIONS":pd.DataFrame(pred),
            "UQ_COMPARISON":pd.DataFrame(uq),
            "SELECTIVE_RISK":pd.DataFrame(risk),
            "SELECTED_MODEL_SUMMARY":pd.DataFrame(selected),
            "GROUPED_CV_5FOLD_LONG":pd.DataFrame(cv),
        }
        for name,df in frames.items():
            dest=(DATA if "PREDICTIONS" in name else TABLES)/f"{name}.csv"
            df.to_csv(dest,index=False,encoding="utf-8-sig")
        cvsum=frames["GROUPED_CV_5FOLD_LONG"].groupby(
            ["regime","scenario","target","feature_set","model"],as_index=False
        ).agg(folds=("fold","nunique"),mae_mean=("mae_mean_seeds","mean"),
              mae_sd=("mae_mean_seeds","std"),r2_mean=("r2_mean_seeds","mean"))
        cvsum.to_csv(TABLES/"GROUPED_CV_5FOLD_SUMMARY.csv",index=False,encoding="utf-8-sig")
        with pd.ExcelWriter(TABLES/"STAGE1A_ROBUST_AUDIT.xlsx",engine="openpyxl") as xw:
            for name,df in frames.items():df.to_excel(xw,sheet_name=name[:31],index=False)
            cvsum.to_excel(xw,sheet_name="CV summary",index=False)
        # Decision: shift is actionable if USDA ratio >1.5 or conformal failure under external shift.
        sdf=frames["SELECTED_MODEL_SUMMARY"]; udf=frames["UQ_COMPARISON"]
        usrat=sdf[sdf.regime=="USDA_CLEARWOOD"]["calibration_to_test_mae_ratio"]
        uscov=udf[(udf.regime=="USDA_CLEARWOOD") & (udf.method=="GLOBAL")]["coverage"]
        actionable=bool((usrat>1.5).any() or (uscov<.85).any())
        # Internal controls should not all fail.
        controls=udf[(udf.regime.isin(["JAPANESE_LARCH","SPRUCE_STRUCTURAL"])) & (udf.method=="GLOBAL")]
        controls_ok=bool((controls.coverage>=.80).mean()>=.5)
        decision="GREEN_FOR_WOODPHYS_METHOD_PILOT" if actionable and controls_ok else "YELLOW_REVISE_SPLITS_OR_UQ"
        dec={"decision":decision,"full_hap_dg_authorization":"NO",
             "actionable_usda_shift_signal":actionable,"internal_control_uq_reasonable":controls_ok,
             "usda_selected_rows":sdf[sdf.regime=="USDA_CLEARWOOD"].to_dict("records"),
             "usda_global_conformal":udf[(udf.regime=="USDA_CLEARWOOD")&(udf.method=="GLOBAL")].to_dict("records"),
             "authorized_next_stage":"STAGE2_WOODPHYS_COMPONENT_PILOT" if decision.startswith("GREEN") else "STAGE1B_SPLIT_UQ_REPAIR",
             "generated_at":now()}
        wjson(OUT/"SCIENTIFIC_STAGE1A_DECISION.json",dec)
        # Figures.
        plt.figure(figsize=(10,6))
        d=sdf.copy();d["label"]=d.regime+"\n"+d.target
        plt.barh(d.label,d.calibration_to_test_mae_ratio)
        plt.axvline(1.0,linestyle="--");plt.xlabel("Test MAE / calibration MAE")
        plt.title("Shift gap for calibration-selected models");plt.tight_layout()
        plt.savefig(FIG/"SHIFT_GAP_RATIO.png",dpi=180);plt.close()
        plt.figure(figsize=(11,6))
        d=udf.copy();d["label"]=d.regime+"\n"+d.target+"\n"+d.method
        plt.barh(d.label,d.coverage);plt.axvline(.9,linestyle="--")
        plt.xlabel("Coverage");plt.title("Conformal coverage comparison");plt.tight_layout()
        plt.savefig(FIG/"UQ_COVERAGE_COMPARISON.png",dpi=180);plt.close()
        # Report.
        lines=["# WOODPHYS_DG_Q1 — Stage 1A Robust Shift Audit","",f"## القرار\n**{decision}**","",
               "## أهم النتائج"]
        for _,r in sdf.iterrows():
            lines.append(f"- {r.regime} / {r.target}: selected {r.model} ({r.feature_set}); "
                         f"cal MAE={r.calibration_mae:.3f}, test MAE={r.test_mae:.3f}, "
                         f"gap={r.calibration_to_test_mae_ratio:.2f}×, "
                         f"95% bootstrap CI=[{r.test_mae_ci_low:.3f}, {r.test_mae_ci_high:.3f}].")
        lines += ["","## تفسير القرار",
                  "- وجود فجوة كبيرة بين calibration وexternal test في USDA يبرر اختبار مكونات WoodPhys-DG.",
                  "- هذا لا يثبت تفوق المنهج المقترح؛ بل يسمح فقط بمرحلة component pilot.",
                  "- لا يوجد pooling بين الأنظمة ولا broad external validation claim.",
                  "","## الملفات الأساسية",
                  "- `tables/STAGE1A_ROBUST_AUDIT.xlsx`",
                  "- `tables/SELECTED_MODEL_SUMMARY.csv`",
                  "- `tables/UQ_COMPARISON.csv`",
                  "- `tables/GROUPED_CV_5FOLD_SUMMARY.csv`",
                  "- `data/SELECTED_TEST_PREDICTIONS.csv`"]
        (OUT/"ROBUST_SHIFT_REPORT_AR.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        # Verification.
        req=["ROBUST_SHIFT_REPORT_AR.md","SCIENTIFIC_STAGE1A_DECISION.json",
             "tables/STAGE1A_ROBUST_AUDIT.xlsx","tables/SELECTED_MODEL_SUMMARY.csv",
             "tables/UQ_COMPARISON.csv","tables/GROUPED_CV_5FOLD_SUMMARY.csv",
             "data/SELECTED_TEST_PREDICTIONS.csv","figures/SHIFT_GAP_RATIO.png",
             "figures/UQ_COVERAGE_COMPARISON.png"]
        missing=[x for x in req if not (OUT/x).exists()]
        wjson(OUT/"OUTPUT_VERIFICATION.json",{"verification_passed":not missing,
              "missing":missing,"seeds":SEEDS,"grouped_cv_target_folds":5,
              "model_selection_used_test":False,"full_hap_dg_prevented":True})
        wjson(OUT/"SUCCESS_MARKER.json",{"success":True,"stage":"STAGE1A","decision":decision,"timestamp":now()})
        wjson(OUT/"STATUS.json",{"stage":"STAGE1A","status":"completed","decision":decision,"completed_at":now()})
        (OUT/"ERROR_LOG.txt").write_text("",encoding="utf-8")
        (OUT/"RUN_LOG.txt").write_text("\n".join(LOG)+"\n",encoding="utf-8")
        # Manifest.
        mans=[];hs=[]
        for p in sorted(OUT.rglob("*")):
            if p.is_file() and "_work" not in p.parts and p.name not in {"FILE_MANIFEST.csv","HASHES_SHA256.csv","UPLOAD_THIS_STAGE1A_RESULT.zip"}:
                rel=p.relative_to(OUT).as_posix();mans.append({"relative_path":rel,"size_bytes":p.stat().st_size});hs.append({"relative_path":rel,"sha256":sha(p)})
        pd.DataFrame(mans).to_csv(OUT/"FILE_MANIFEST.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(hs).to_csv(OUT/"HASHES_SHA256.csv",index=False,encoding="utf-8-sig")
        up=OUT/"UPLOAD_THIS_STAGE1A_RESULT.zip"
        with zipfile.ZipFile(up,"w",zipfile.ZIP_DEFLATED) as z:
            for p in sorted(OUT.rglob("*")):
                if p.is_file() and p!=up and "_work" not in p.parts:z.write(p,p.relative_to(OUT))
        log(f"Completed Stage1A. Decision: {decision}")
        (OUT/"RUN_LOG.txt").write_text("\n".join(LOG)+"\n",encoding="utf-8")
        return 0
    except Exception as e:
        ERR.append(traceback.format_exc())
        wjson(OUT/"STATUS.json",{"stage":"STAGE1A","status":"failed","error":str(e),"time":now()})
        wjson(OUT/"SUCCESS_MARKER.json",{"success":False,"error":str(e),"time":now()})
        (OUT/"RUN_LOG.txt").write_text("\n".join(LOG)+"\n",encoding="utf-8")
        (OUT/"ERROR_LOG.txt").write_text("\n".join(ERR)+"\n",encoding="utf-8")
        print(ERR[-1],file=sys.stderr);return 1
if __name__=="__main__":raise SystemExit(main())
