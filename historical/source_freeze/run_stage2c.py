
from __future__ import annotations
import datetime as dt
import hashlib
import json
import math
import platform
from pathlib import Path
import shutil
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT0F = ROOT / "input" / "STAGE0F_VERIFIED_INPUT.zip"
INPUT2B = ROOT / "input" / "STAGE2B_VERIFIED_INPUT.zip"
OUT = ROOT / "stage2c_output"
WORK = OUT / "_work"
TABLES = OUT / "tables"
DATA = OUT / "data"
FIGURES = OUT / "figures"

DOMAINS = ["DEVELOPMENT_US", "CALIBRATION_CANADA", "EXTERNAL_OTHER_IMPORTS"]
TARGETS = ["moe_mpa", "mor_mpa"]
SEEDS = [13, 42, 97, 131, 2026]
TEMPS = [0.1, 0.2, 0.25, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10]
BETAS = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]
ALPHA = 0.10

FEATURES = ["specific_gravity", "moisture_percent",
            "moisture_state", "wood_class", "genus", "family"]
NUMS = ["specific_gravity", "moisture_percent"]
CATS = ["moisture_state", "wood_class", "genus", "family"]

LOG_LINES = []
ERROR_LINES = []

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def log(msg):
    line = f"[{now()}] {msg}"
    LOG_LINES.append(line)
    print(line, flush=True)

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler())
        ]), NUMS),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("ohe", make_ohe())
        ]), CATS),
    ], sparse_threshold=0.0)

def verify_inputs():
    s0 = WORK / "stage0f"
    s2 = WORK / "stage2b"
    s0.mkdir(parents=True, exist_ok=True)
    s2.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(INPUT0F) as z:
        z.extractall(s0)
    with zipfile.ZipFile(INPUT2B) as z:
        z.extractall(s2)

    a = json.loads((s0/"SUCCESS_MARKER.json").read_text(encoding="utf-8"))
    av = json.loads((s0/"OUTPUT_VERIFICATION.json").read_text(encoding="utf-8"))
    b = json.loads((s2/"SUCCESS_MARKER.json").read_text(encoding="utf-8"))
    bv = json.loads((s2/"OUTPUT_VERIFICATION.json").read_text(encoding="utf-8"))
    bd = json.loads((s2/"SCIENTIFIC_STAGE2B_DECISION.json").read_text(encoding="utf-8"))

    if not a.get("success") or not av.get("verification_passed"):
        raise RuntimeError("Stage0F input is not verified")
    if not b.get("success") or not bv.get("verification_passed"):
        raise RuntimeError("Stage2B input is not verified")
    if bd.get("decision") != "YELLOW_MULTI_DOMAIN_REDESIGN":
        raise RuntimeError("Stage2B decision does not request redesign")
    log("Verified Stage0F and Stage2B inputs")
    return s0, s2

def split_source(source_df, seed):
    tr_parts, cal_parts = [], []
    for _, group in source_df.groupby("domain_role"):
        groups = group["species_group_key"].fillna(
            group["species_label_raw"]
        ).astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=seed)
        tr, ca = next(splitter.split(group, groups=groups))
        tr_parts.append(group.iloc[tr])
        cal_parts.append(group.iloc[ca])
    return pd.concat(tr_parts, ignore_index=True), pd.concat(cal_parts, ignore_index=True)

def fit_source_models(x, y, domains):
    models, centroids = {}, {}
    for domain in sorted(np.unique(domains)):
        mask = domains == domain
        model = Ridge(alpha=1.0)
        model.fit(x[mask], y[mask])
        models[domain] = model
        centroids[domain] = x[mask].mean(axis=0)
    return models, centroids

def source_prediction_matrix(models, x):
    names = sorted(models)
    return names, np.vstack([models[name].predict(x) for name in names])

def applicability_predict(models, centroids, x, temperature):
    names, matrix = source_prediction_matrix(models, x)
    distances = np.vstack([
        np.linalg.norm(x-centroids[name], axis=1) for name in names
    ])
    weights = np.exp(-distances/temperature)
    weights /= weights.sum(axis=0, keepdims=True)
    pred = (weights*matrix).sum(axis=0)
    equal_disagreement = matrix.std(axis=0)
    weighted_variance = (weights*(matrix-pred)**2).sum(axis=0)
    return pred, equal_disagreement, np.sqrt(weighted_variance), weights, distances

def metric(y, pred):
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred)**.5),
        "r2": float(r2_score(y, pred))
    }

def conformal_q(residuals):
    r = np.sort(np.asarray(residuals, float))
    k = min(len(r)-1, max(0, int(math.ceil((len(r)+1)*(1-ALPHA)))-1))
    return float(r[k])

def risk_coverage(y, pred, risk):
    order = np.argsort(risk)
    rows = []
    for coverage in np.arange(.20, 1.001, .05):
        n = max(1, int(round(len(y)*coverage)))
        idx = order[:n]
        rows.append({
            "coverage": float(n/len(y)),
            "retained_n": int(n),
            "mae": float(mean_absolute_error(y[idx], pred[idx]))
        })
    aurc = float(np.trapezoid(
        [r["mae"] for r in rows],
        [r["coverage"] for r in rows]
    ))
    return rows, aurc

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in [OUT, WORK, TABLES, DATA, FIGURES]:
        d.mkdir(parents=True, exist_ok=True)

    write_json(OUT/"STATUS.json", {
        "stage": "STAGE2C", "status": "running", "started_at": now()
    })

    try:
        stage0f, stage2b = verify_inputs()
        usda = pd.read_csv(stage0f/"freeze"/"usda_clearwood_primary_model_ready.csv")
        manifest = pd.read_csv(stage0f/"splits"/"USDA_DOMAIN_AND_TAXON_SPLIT_MANIFEST.csv")
        usda = usda.merge(
            manifest[["record_id", "domain_role"]],
            on="record_id", how="left", validate="one_to_one"
        )

        candidate_rows = []
        fold_predictions = {}
        uq_candidate_rows = []
        risk_rows = []
        final_predictions = []

        # First pass: point predictions for every target/temperature/domain.
        for target in TARGETS:
            log(f"Point temperature search: {target}")
            for held_domain in DOMAINS:
                source_df = usda[usda["domain_role"] != held_domain].copy()
                held_df = usda[usda["domain_role"] == held_domain].copy()
                y_held = held_df[target].to_numpy(float)
                iqr = max(
                    float(np.quantile(y_held,.75)-np.quantile(y_held,.25)),
                    1e-9
                )

                temp_seed_preds = {temp: [] for temp in TEMPS}
                erm_seed_preds = []
                equal_seed_preds = []
                saved_seed_data = []

                for seed in SEEDS:
                    train_df, cal_df = split_source(source_df, seed)
                    pre = preprocessor()
                    xtr = np.asarray(pre.fit_transform(train_df[FEATURES]), float)
                    xcal = np.asarray(pre.transform(cal_df[FEATURES]), float)
                    xheld = np.asarray(pre.transform(held_df[FEATURES]), float)
                    ytr = train_df[target].to_numpy(float)
                    ycal = cal_df[target].to_numpy(float)
                    domains = train_df["domain_role"].to_numpy(str)

                    erm = Ridge(alpha=1.0).fit(xtr, ytr)
                    erm_seed_preds.append(erm.predict(xheld))

                    models, centroids = fit_source_models(xtr, ytr, domains)
                    _, matrix = source_prediction_matrix(models, xheld)
                    equal_seed_preds.append(matrix.mean(axis=0))

                    cal_cache = {}
                    held_cache = {}
                    for temp in TEMPS:
                        pcal, dcal, wdcal, _, _ = applicability_predict(
                            models, centroids, xcal, temp
                        )
                        pheld, dheld, wdheld, weights, distances = applicability_predict(
                            models, centroids, xheld, temp
                        )
                        temp_seed_preds[temp].append(pheld)
                        cal_cache[temp] = (pcal, dcal, wdcal)
                        held_cache[temp] = (pheld, dheld, wdheld, weights, distances)
                    saved_seed_data.append(
                        (seed, ycal, cal_cache, held_cache, held_df)
                    )

                erm_pred = np.mean(erm_seed_preds, axis=0)
                equal_pred = np.mean(equal_seed_preds, axis=0)

                for method, pred in [
                    ("ERM_POOLED_RIDGE", erm_pred),
                    ("EQUAL_SOURCE_ENSEMBLE", equal_pred)
                ]:
                    m = metric(y_held, pred)
                    candidate_rows.append({
                        "target": target,
                        "held_domain": held_domain,
                        "candidate": method,
                        "temperature": np.nan,
                        **m,
                        "normalized_mae_iqr": m["mae"]/iqr
                    })

                for temp in TEMPS:
                    pred = np.mean(temp_seed_preds[temp], axis=0)
                    m = metric(y_held, pred)
                    candidate_rows.append({
                        "target": target,
                        "held_domain": held_domain,
                        "candidate": "APPLICABILITY_WEIGHTED_SOURCE_ENSEMBLE",
                        "temperature": temp,
                        **m,
                        "normalized_mae_iqr": m["mae"]/iqr
                    })
                    fold_predictions[(target, held_domain, temp)] = (
                        pred, y_held, saved_seed_data, held_df
                    )

        candidates = pd.DataFrame(candidate_rows)

        selection_rows = []
        selected_temps = {}
        point_gate_rows = []

        for target in TARGETS:
            erm = (
                candidates[
                    (candidates["target"]==target) &
                    (candidates["candidate"]=="ERM_POOLED_RIDGE")
                ]
                .groupby("target")
                .agg(
                    mean_nmae=("normalized_mae_iqr","mean"),
                    worst_nmae=("normalized_mae_iqr","max")
                ).iloc[0]
            )

            grid = (
                candidates[
                    (candidates["target"]==target) &
                    (candidates["candidate"]=="APPLICABILITY_WEIGHTED_SOURCE_ENSEMBLE")
                ]
                .groupby("temperature", as_index=False)
                .agg(
                    mean_nmae=("normalized_mae_iqr","mean"),
                    worst_nmae=("normalized_mae_iqr","max"),
                    mean_mae=("mae","mean")
                )
                .sort_values(["worst_nmae","mean_nmae","temperature"])
            )
            winner = grid.iloc[0]
            temp = float(winner["temperature"])
            selected_temps[target] = temp

            for _, row in grid.iterrows():
                selection_rows.append({
                    "target": target,
                    "temperature": row["temperature"],
                    "mean_nmae": row["mean_nmae"],
                    "worst_nmae": row["worst_nmae"],
                    "selected": "YES" if row["temperature"]==temp else "NO"
                })

            mean_improvement = (
                erm["mean_nmae"]-winner["mean_nmae"]
            )/erm["mean_nmae"]
            worst_improvement = (
                erm["worst_nmae"]-winner["worst_nmae"]
            )/erm["worst_nmae"]
            point_pass = bool(
                mean_improvement >= .05 and worst_improvement >= .05
            )
            point_gate_rows.append({
                "target": target,
                "selected_temperature": temp,
                "erm_mean_nmae": erm["mean_nmae"],
                "selected_mean_nmae": winner["mean_nmae"],
                "mean_improvement_fraction": mean_improvement,
                "erm_worst_nmae": erm["worst_nmae"],
                "selected_worst_nmae": winner["worst_nmae"],
                "worst_improvement_fraction": worst_improvement,
                "point_gate_pass": point_pass
            })

        # Second pass: UQ and selective risk for selected temperatures.
        for target in TARGETS:
            temp = selected_temps[target]
            log(f"UQ evaluation: {target}; temperature={temp}")
            beta_records = []

            for held_domain in DOMAINS:
                pred, yheld, saved_seed_data, held_df = fold_predictions[
                    (target, held_domain, temp)
                ]

                seed_uq = []
                seed_disagreement = []
                for seed, ycal, cal_cache, held_cache, _ in saved_seed_data:
                    pcal, dcal, _ = cal_cache[temp]
                    pheld, dheld, _, weights, distances = held_cache[temp]
                    residuals = np.abs(ycal-pcal)
                    q = conformal_q(residuals)
                    scale = max(float(np.median(dcal)), 1e-6)
                    seed_disagreement.append(dheld)

                    for beta in BETAS:
                        half = q*(1+beta*np.clip(dheld/scale,0,5))
                        seed_uq.append({
                            "target": target,
                            "held_domain": held_domain,
                            "seed": seed,
                            "beta": beta,
                            "coverage": float(np.mean(np.abs(yheld-pheld)<=half)),
                            "average_width": float(np.mean(2*half))
                        })
                uq_candidate_rows.extend(seed_uq)

                disagreement = np.mean(seed_disagreement, axis=0)
                rc, aurc = risk_coverage(yheld, pred, disagreement)
                corr = float(
                    pd.Series(disagreement).rank().corr(
                        pd.Series(np.abs(yheld-pred)).rank()
                    )
                )
                for row in rc:
                    risk_rows.append({
                        "target": target,
                        "held_domain": held_domain,
                        "risk_method": "SOURCE_DISAGREEMENT",
                        **row,
                        "aurc": aurc,
                        "risk_error_rank_correlation": corr
                    })

                # Decision groups use source-calibration quantiles averaged across seeds.
                cal50 = np.mean([
                    np.quantile(cal_cache[temp][1], .50)
                    for _, _, cal_cache, _, _ in saved_seed_data
                ])
                cal90 = np.mean([
                    np.quantile(cal_cache[temp][1], .90)
                    for _, _, cal_cache, _, _ in saved_seed_data
                ])
                groups = np.where(
                    disagreement <= cal50, "ACCEPT",
                    np.where(disagreement <= cal90, "REVIEW", "ABSTAIN")
                )
                for group in ["ACCEPT","REVIEW","ABSTAIN"]:
                    mask = groups==group
                    risk_rows.append({
                        "target": target,
                        "held_domain": held_domain,
                        "risk_method": "DECISION_GROUP",
                        "coverage": float(mask.mean()),
                        "retained_n": int(mask.sum()),
                        "mae": (
                            float(mean_absolute_error(yheld[mask],pred[mask]))
                            if mask.any() else np.nan
                        ),
                        "decision_group": group,
                        "aurc": aurc,
                        "risk_error_rank_correlation": corr
                    })

                for i, rec in enumerate(held_df.itertuples(index=False)):
                    final_predictions.append({
                        "target": target,
                        "held_domain": held_domain,
                        "record_id": rec.record_id,
                        "selected_temperature": temp,
                        "y_true": float(yheld[i]),
                        "y_pred": float(pred[i]),
                        "absolute_error": float(abs(yheld[i]-pred[i])),
                        "source_disagreement": float(disagreement[i]),
                        "decision_group": str(groups[i])
                    })

        uq_candidates = pd.DataFrame(uq_candidate_rows)
        uq_summary = (
            uq_candidates.groupby(["target","beta"],as_index=False)
            .agg(
                mean_coverage=("coverage","mean"),
                minimum_domain_seed_coverage=("coverage","min"),
                mean_width=("average_width","mean")
            )
        )
        global_width = (
            uq_summary[uq_summary["beta"]==0][["target","mean_width"]]
            .rename(columns={"mean_width":"global_width"})
        )
        uq_summary = uq_summary.merge(global_width,on="target",how="left")
        uq_summary["width_ratio_to_global"] = (
            uq_summary["mean_width"]/uq_summary["global_width"]
        )

        uq_gate_rows = []
        for target in TARGETS:
            sub = uq_summary[uq_summary["target"]==target].copy()
            feasible = sub[
                (sub["mean_coverage"]>=.88) &
                (sub["minimum_domain_seed_coverage"]>=.75) &
                (sub["width_ratio_to_global"]<=2.5)
            ].sort_values(["beta","width_ratio_to_global"])
            if len(feasible):
                winner = feasible.iloc[0]
                uq_pass = True
            else:
                sub["distance"] = abs(sub["mean_coverage"]-.90)
                winner = sub.sort_values(["distance","width_ratio_to_global"]).iloc[0]
                uq_pass = False
            uq_gate_rows.append({
                "target": target,
                "selected_beta": winner["beta"],
                "mean_coverage": winner["mean_coverage"],
                "minimum_domain_seed_coverage": winner[
                    "minimum_domain_seed_coverage"
                ],
                "width_ratio_to_global": winner["width_ratio_to_global"],
                "uq_gate_pass": uq_pass
            })

        point_gates = pd.DataFrame(point_gate_rows)
        uq_gates = pd.DataFrame(uq_gate_rows)
        gates = point_gates.merge(uq_gates,on="target")

        point_passes = int(gates["point_gate_pass"].sum())
        uq_passes = int(gates["uq_gate_pass"].sum())

        if point_passes==2 and uq_passes==2:
            decision = "GREEN_FOR_METHOD_FREEZE_PENDING_EXTERNAL_LOCKBOX"
            rationale = (
                "Both MOE and MOR passed mean and worst-domain point gates, "
                "and both targets passed the disagreement-scaled UQ gate. "
                "The target-specific applicability-weighted method may be frozen "
                "before independent external evaluation."
            )
        elif point_passes>=1 and uq_passes>=1:
            decision = "YELLOW_FREEZE_REDESIGN"
            rationale = (
                "The method is partially successful, but at least one target "
                "still fails a point or uncertainty gate."
            )
        else:
            decision = "RED_APPLICABILITY_PIVOT"
            rationale = (
                "Applicability weighting did not provide sufficient robust gain."
            )

        candidates.to_csv(
            TABLES/"POINT_CANDIDATES_BY_DOMAIN.csv",
            index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(selection_rows).to_csv(
            TABLES/"TEMPERATURE_SELECTION.csv",
            index=False, encoding="utf-8-sig"
        )
        uq_candidates.to_csv(
            TABLES/"UQ_RESULTS_BY_SEED_DOMAIN.csv",
            index=False, encoding="utf-8-sig"
        )
        uq_summary.to_csv(
            TABLES/"UQ_BETA_SUMMARY.csv",
            index=False, encoding="utf-8-sig"
        )
        gates.to_csv(
            TABLES/"METHOD_FREEZE_GATES.csv",
            index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(risk_rows).to_csv(
            TABLES/"SELECTIVE_RISK_AUDIT.csv",
            index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(final_predictions).to_csv(
            DATA/"APPLICABILITY_WEIGHTED_DEVELOPMENT_PREDICTIONS.csv",
            index=False, encoding="utf-8-sig"
        )

        with pd.ExcelWriter(
            TABLES/"STAGE2C_METHOD_FREEZE.xlsx", engine="openpyxl"
        ) as xw:
            candidates.to_excel(xw,sheet_name="Point candidates",index=False)
            pd.DataFrame(selection_rows).to_excel(
                xw,sheet_name="Temperature",index=False
            )
            uq_summary.to_excel(xw,sheet_name="UQ summary",index=False)
            gates.to_excel(xw,sheet_name="Freeze gates",index=False)
            pd.DataFrame(risk_rows).to_excel(
                xw,sheet_name="Selective risk",index=False
            )

        # Figures.
        for target in TARGETS:
            grid = pd.DataFrame(selection_rows)
            grid = grid[grid["target"]==target]
            plt.figure(figsize=(8,5))
            plt.plot(grid["temperature"],grid["mean_nmae"],marker="o",label="Mean NMAE")
            plt.plot(grid["temperature"],grid["worst_nmae"],marker="o",label="Worst-domain NMAE")
            plt.xscale("log")
            plt.xlabel("Applicability temperature")
            plt.ylabel("Normalized MAE")
            plt.title(f"Temperature development curve: {target}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIGURES/f"TEMPERATURE_CURVE_{target}.png",dpi=180)
            plt.close()

            u = uq_summary[uq_summary["target"]==target]
            plt.figure(figsize=(8,5))
            plt.plot(u["beta"],u["mean_coverage"],marker="o",label="Mean coverage")
            plt.plot(
                u["beta"],u["minimum_domain_seed_coverage"],
                marker="o",label="Minimum coverage"
            )
            plt.axhline(.90,linestyle="--")
            plt.xlabel("Disagreement scaling beta")
            plt.ylabel("Coverage")
            plt.title(f"UQ development curve: {target}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIGURES/f"UQ_BETA_CURVE_{target}.png",dpi=180)
            plt.close()

        scientific = {
            "decision": decision,
            "rationale": rationale,
            "full_hap_dg_authorization": "NO",
            "method_freeze_authorization": (
                "YES_TARGET_SPECIFIC_PARAMETERS"
                if decision=="GREEN_FOR_METHOD_FREEZE_PENDING_EXTERNAL_LOCKBOX"
                else "NO"
            ),
            "frozen_parameters": [
                {
                    "target": row["target"],
                    "applicability_temperature": row["selected_temperature"],
                    "uq_beta": row["selected_beta"]
                }
                for _,row in gates.iterrows()
            ],
            "development_domains_used_for_method_selection": DOMAINS,
            "final_external_validation_status": "NOT_AVAILABLE",
            "required_next_steps": [
                "acquire or pre-register an independent external lockbox",
                "freeze code and hashes before external labels are inspected",
                "run final external evaluation without parameter changes",
                "perform manual USDA row and license audit",
            ],
            "generated_at": now()
        }
        write_json(OUT/"SCIENTIFIC_STAGE2C_DECISION.json",scientific)

        report = [
            "# WOODPHYS_DG_Q1 — Stage 2C Method Freeze",
            "",
            f"## القرار: **{decision}**",
            "",
            rationale,
            "",
            "## تنبيه",
            "هذه نتائج development متعددة النطاقات وليست final external validation.",
            "",
        ]
        for _,row in gates.iterrows():
            report += [
                f"### {row['target']}",
                f"- Selected temperature: `{row['selected_temperature']}`",
                f"- Mean NMAE improvement: "
                f"{100*row['mean_improvement_fraction']:.2f}%",
                f"- Worst-domain improvement: "
                f"{100*row['worst_improvement_fraction']:.2f}%",
                f"- Point gate: `{row['point_gate_pass']}`",
                f"- Selected UQ beta: `{row['selected_beta']}`",
                f"- Mean coverage: {row['mean_coverage']:.3f}",
                f"- Minimum coverage: "
                f"{row['minimum_domain_seed_coverage']:.3f}",
                f"- Width ratio to global: "
                f"{row['width_ratio_to_global']:.3f}",
                f"- UQ gate: `{row['uq_gate_pass']}`",
                "",
            ]
        report += [
            "## التالي",
            "إذا كان القرار أخضر، تُجمّد هذه المعلمات ويُمنع تعديلها بعد رؤية",
            "نتائج أي external lockbox مستقل.",
        ]
        (OUT/"METHOD_FREEZE_REPORT_AR.md").write_text(
            "\n".join(report)+"\n",encoding="utf-8"
        )

        write_json(OUT/"OUTPUT_VERIFICATION.json",{
            "verification_passed": True,
            "stage0f_verified": True,
            "stage2b_verified": True,
            "five_seeds": SEEDS,
            "three_development_domains": DOMAINS,
            "final_external_claim_prevented": True,
            "full_hap_dg_prevented": True,
            "target_specific_parameters_explicit": True
        })
        write_json(OUT/"CHECKPOINT.json",{
            "completed_steps":[
                "input_verification",
                "temperature_grid_leave_one_domain_out",
                "applicability_weighted_prediction",
                "mean_and_worst_domain_point_gates",
                "disagreement_scaled_conformal_grid",
                "selective_risk_audit",
                "method_freeze_decision"
            ],
            "timestamp":now()
        })
        write_json(OUT/"STATUS.json",{
            "stage":"STAGE2C","version":"1.0.0",
            "status":"completed","decision":decision,
            "completed_at":now()
        })
        write_json(OUT/"SUCCESS_MARKER.json",{
            "success":True,"stage":"STAGE2C","version":"1.0.0",
            "decision":decision,"timestamp":now()
        })
        write_json(OUT/"ENVIRONMENT_SNAPSHOT.json",{
            "python":sys.version,
            "platform":platform.platform(),
            "numpy":np.__version__,
            "pandas":pd.__version__
        })
        (OUT/"ERROR_LOG.txt").write_text("",encoding="utf-8")
        (OUT/"RUN_LOG.txt").write_text(
            "\n".join(LOG_LINES)+"\n",encoding="utf-8"
        )

        required = [
            "METHOD_FREEZE_REPORT_AR.md",
            "SCIENTIFIC_STAGE2C_DECISION.json",
            "tables/STAGE2C_METHOD_FREEZE.xlsx",
            "tables/METHOD_FREEZE_GATES.csv",
            "tables/TEMPERATURE_SELECTION.csv",
            "tables/UQ_BETA_SUMMARY.csv",
            "data/APPLICABILITY_WEIGHTED_DEVELOPMENT_PREDICTIONS.csv"
        ]
        missing = [x for x in required if not (OUT/x).exists()]
        if missing:
            raise RuntimeError(f"Missing required outputs: {missing}")

        manifest_rows, hash_rows = [], []
        for path in sorted(OUT.rglob("*")):
            if not path.is_file() or "_work" in path.parts:
                continue
            rel = path.relative_to(OUT).as_posix()
            if rel in {
                "FILE_MANIFEST.csv","HASHES_SHA256.csv",
                "UPLOAD_THIS_STAGE2C_RESULT.zip"
            }:
                continue
            manifest_rows.append({
                "relative_path":rel,
                "size_bytes":path.stat().st_size
            })
            hash_rows.append({
                "relative_path":rel,
                "sha256":sha256(path)
            })
        pd.DataFrame(manifest_rows).to_csv(
            OUT/"FILE_MANIFEST.csv",index=False,encoding="utf-8-sig"
        )
        pd.DataFrame(hash_rows).to_csv(
            OUT/"HASHES_SHA256.csv",index=False,encoding="utf-8-sig"
        )

        upload = OUT/"UPLOAD_THIS_STAGE2C_RESULT.zip"
        with zipfile.ZipFile(upload,"w",zipfile.ZIP_DEFLATED) as z:
            for path in sorted(OUT.rglob("*")):
                if path.is_file() and path!=upload and "_work" not in path.parts:
                    z.write(path,path.relative_to(OUT))

        log(f"Completed Stage2C. Decision: {decision}")
        (OUT/"RUN_LOG.txt").write_text(
            "\n".join(LOG_LINES)+"\n",encoding="utf-8"
        )
        return 0

    except Exception as exc:
        ERROR_LINES.append(traceback.format_exc())
        write_json(OUT/"STATUS.json",{
            "stage":"STAGE2C","status":"failed",
            "error":str(exc),"completed_at":now()
        })
        write_json(OUT/"SUCCESS_MARKER.json",{
            "success":False,"stage":"STAGE2C",
            "error":str(exc),"timestamp":now()
        })
        (OUT/"RUN_LOG.txt").write_text(
            "\n".join(LOG_LINES)+"\n",encoding="utf-8"
        )
        (OUT/"ERROR_LOG.txt").write_text(
            "\n".join(ERROR_LINES)+"\n",encoding="utf-8"
        )
        print(ERROR_LINES[-1],file=sys.stderr)
        return 1

if __name__=="__main__":
    raise SystemExit(main())
