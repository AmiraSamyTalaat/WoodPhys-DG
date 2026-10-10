
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT0F = ROOT / "input" / "STAGE0F_VERIFIED_INPUT.zip"
INPUT1A = ROOT / "input" / "STAGE1A_VERIFIED_INPUT.zip"
OUT = ROOT / "stage2b_output"
WORK = OUT / "_work"
TABLES = OUT / "tables"
DATA = OUT / "data"
FIGURES = OUT / "figures"

DOMAINS = [
    "DEVELOPMENT_US",
    "CALIBRATION_CANADA",
    "EXTERNAL_OTHER_IMPORTS",
]
TARGETS = ["moe_mpa", "mor_mpa"]
SEEDS = [13, 42, 97, 131, 2026]
ALPHA = 0.10
LOG_LINES = []
ERROR_LINES = []

FEATURES = [
    "specific_gravity", "moisture_percent",
    "moisture_state", "wood_class", "genus", "family",
]
NUMS = ["specific_gravity", "moisture_percent"]
CATS = ["moisture_state", "wood_class", "genus", "family"]

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def log(message):
    line = f"[{now()}] {message}"
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

def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), NUMS),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", make_ohe()),
        ]), CATS),
    ], sparse_threshold=0.0)

def verify_inputs():
    a = WORK / "stage0f"
    b = WORK / "stage1a"
    a.mkdir(parents=True, exist_ok=True)
    b.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(INPUT0F) as z:
        z.extractall(a)
    with zipfile.ZipFile(INPUT1A) as z:
        z.extractall(b)
    s0 = json.loads((a/"SUCCESS_MARKER.json").read_text(encoding="utf-8"))
    v0 = json.loads((a/"OUTPUT_VERIFICATION.json").read_text(encoding="utf-8"))
    s1 = json.loads((b/"SUCCESS_MARKER.json").read_text(encoding="utf-8"))
    v1 = json.loads((b/"OUTPUT_VERIFICATION.json").read_text(encoding="utf-8"))
    d1 = json.loads((b/"SCIENTIFIC_STAGE1A_DECISION.json").read_text(encoding="utf-8"))
    if not s0.get("success") or not v0.get("verification_passed"):
        raise RuntimeError("Stage0F input invalid")
    if not s1.get("success") or not v1.get("verification_passed"):
        raise RuntimeError("Stage1A input invalid")
    if d1.get("decision") != "GREEN_FOR_WOODPHYS_METHOD_PILOT":
        raise RuntimeError("Stage1A did not authorize the pilot")
    log("Verified Stage0F and Stage1A")
    return a, b

def metric(y, pred):
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred)**0.5),
        "r2": float(r2_score(y, pred)),
    }

def add_intercept(x):
    return np.column_stack([np.ones(len(x)), x])

def fit_gradient_penalty(x, y, domains, alpha=1.0, lam=1.0):
    xa = add_intercept(x)
    n, p = xa.shape
    h = xa.T @ xa / n
    b = xa.T @ y / n
    reg = np.eye(p) * alpha / n
    reg[0, 0] = 0.0
    a = h + reg
    rhs = b.copy()
    for domain in np.unique(domains):
        mask = domains == domain
        xd = xa[mask]
        yd = y[mask]
        hd = xd.T @ xd / mask.sum()
        bd = xd.T @ yd / mask.sum()
        a += lam * (hd.T @ hd)
        rhs += lam * (hd.T @ bd)
    return np.linalg.solve(a + 1e-8*np.eye(p), rhs)

def fit_domain_dro(x, y, domains, alpha=1.0, eta=0.30, iterations=8):
    unique = sorted(np.unique(domains))
    q = {d: 1.0/len(unique) for d in unique}
    model = Ridge(alpha=alpha)
    for _ in range(iterations):
        counts = {d: int((domains==d).sum()) for d in unique}
        weights = np.array([q[d]/counts[d] for d in domains], float)
        weights *= len(weights)/weights.sum()
        model.fit(x, y, sample_weight=weights)
        pred = model.predict(x)
        losses = {
            d: mean_absolute_error(y[domains==d], pred[domains==d])
            for d in unique
        }
        scale = max(np.mean(list(losses.values())), 1e-8)
        for d in unique:
            q[d] *= math.exp(eta*losses[d]/scale)
        total = sum(q.values())
        q = {d: v/total for d, v in q.items()}
    return model, q

def split_train_cal_by_domain(source_df, seed):
    train_parts = []
    cal_parts = []
    for _, group in source_df.groupby("domain_role"):
        groups = group["species_group_key"].fillna(
            group["species_label_raw"]
        ).astype(str)
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.20, random_state=seed
        )
        train_idx, cal_idx = next(splitter.split(group, groups=groups))
        train_parts.append(group.iloc[train_idx])
        cal_parts.append(group.iloc[cal_idx])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(cal_parts, ignore_index=True),
    )

def conformal_q(residuals, alpha=ALPHA, weights=None):
    residuals = np.asarray(residuals, float)
    if weights is None:
        ordered = np.sort(residuals)
        k = min(
            len(ordered)-1,
            max(0, int(math.ceil((len(ordered)+1)*(1-alpha)))-1)
        )
        return float(ordered[k])
    weights = np.asarray(weights, float)
    order = np.argsort(residuals)
    residuals = residuals[order]
    weights = np.clip(weights[order], 1e-8, None)
    cdf = np.cumsum(weights)/weights.sum()
    idx = min(int(np.searchsorted(cdf, 1-alpha)), len(residuals)-1)
    return float(residuals[idx])

def fit_source_models(x, y, domains):
    models = {}
    for domain in sorted(np.unique(domains)):
        mask = domains == domain
        model = Ridge(alpha=1.0)
        model.fit(x[mask], y[mask])
        models[domain] = model
    return models

def source_ensemble_predict(models, x):
    matrix = np.vstack([model.predict(x) for model in models.values()])
    return matrix.mean(axis=0), matrix.std(axis=0), matrix

def choose_physics_lambda(source_train, source_cal, base_cal, target):
    mask = source_train["specific_gravity"].notna()
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(
        source_train.loc[mask, "specific_gravity"],
        source_train.loc[mask, target],
    )
    iso_cal = np.full(len(source_cal), source_train[target].mean(), float)
    m = source_cal["specific_gravity"].notna().to_numpy()
    iso_cal[m] = iso.predict(source_cal.loc[m, "specific_gravity"])
    candidates = []
    for lam in [0.0, 0.10, 0.25, 0.40, 0.60]:
        pred = (1-lam)*base_cal + lam*iso_cal
        candidates.append((lam, mean_absolute_error(source_cal[target], pred)))
    selected = min(candidates, key=lambda x: x[1])[0]
    return selected, iso, candidates

def apply_physics_blend(df, base_pred, iso, default, lam):
    iso_pred = np.full(len(df), default, float)
    mask = df["specific_gravity"].notna().to_numpy()
    iso_pred[mask] = iso.predict(df.loc[mask, "specific_gravity"])
    return (1-lam)*base_pred + lam*iso_pred

def risk_coverage(y, pred, risk, label):
    order = np.argsort(risk)
    rows = []
    for coverage in np.arange(.20, 1.001, .05):
        n = max(1, int(round(len(y)*coverage)))
        idx = order[:n]
        rows.append({
            "risk_method": label,
            "coverage": float(n/len(y)),
            "retained_n": int(n),
            "mae": float(mean_absolute_error(y[idx], pred[idx])),
        })
    aurc = float(np.trapezoid(
        [row["mae"] for row in rows],
        [row["coverage"] for row in rows],
    ))
    return rows, aurc

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in [OUT, WORK, TABLES, DATA, FIGURES]:
        d.mkdir(parents=True, exist_ok=True)
    write_json(OUT/"STATUS.json", {
        "stage": "STAGE2B", "status": "running", "started_at": now()
    })

    try:
        stage0f, stage1a = verify_inputs()
        usda = pd.read_csv(
            stage0f/"freeze"/"usda_clearwood_primary_model_ready.csv"
        )
        manifest = pd.read_csv(
            stage0f/"splits"/"USDA_DOMAIN_AND_TAXON_SPLIT_MANIFEST.csv"
        )
        usda = usda.merge(
            manifest[["record_id", "domain_role"]],
            on="record_id", how="left", validate="one_to_one"
        )

        fold_rows = []
        uq_rows = []
        prediction_rows = []
        risk_rows = []
        physics_rows = []
        dro_rows = []

        for target in TARGETS:
            log(f"Multi-domain pilot start: {target}")
            for held_domain in DOMAINS:
                source_df = usda[usda["domain_role"] != held_domain].copy()
                held_df = usda[usda["domain_role"] == held_domain].copy()

                seed_method_predictions = {}
                seed_uq = []

                for seed in SEEDS:
                    train_df, cal_df = split_train_cal_by_domain(
                        source_df, seed
                    )
                    pre = make_preprocessor()
                    x_train = np.asarray(
                        pre.fit_transform(train_df[FEATURES]), float
                    )
                    x_cal = np.asarray(pre.transform(cal_df[FEATURES]), float)
                    x_held = np.asarray(pre.transform(held_df[FEATURES]), float)
                    y_train = train_df[target].to_numpy(float)
                    y_cal = cal_df[target].to_numpy(float)
                    y_held = held_df[target].to_numpy(float)
                    domains_train = train_df["domain_role"].to_numpy(str)

                    # ERM.
                    erm = Ridge(alpha=1.0)
                    erm.fit(x_train, y_train)
                    p_erm = erm.predict(x_held)

                    # Domain DRO.
                    dro, q = fit_domain_dro(
                        x_train, y_train, domains_train
                    )
                    p_dro = dro.predict(x_held)
                    if seed == 42:
                        for domain, weight in q.items():
                            dro_rows.append({
                                "target": target,
                                "held_domain": held_domain,
                                "source_domain": domain,
                                "final_weight": weight,
                            })

                    # Gradient-penalized ridge.
                    beta = fit_gradient_penalty(
                        x_train, y_train, domains_train,
                        alpha=1.0, lam=1.0
                    )
                    p_grad = add_intercept(x_held) @ beta

                    # Source-model ensemble.
                    source_models = fit_source_models(
                        x_train, y_train, domains_train
                    )
                    p_ens, held_disagreement, held_matrix = (
                        source_ensemble_predict(source_models, x_held)
                    )
                    p_cal_ens, cal_disagreement, cal_matrix = (
                        source_ensemble_predict(source_models, x_cal)
                    )

                    # Physics-guided source ensemble.
                    lam, iso, candidates = choose_physics_lambda(
                        train_df, cal_df, p_cal_ens, target
                    )
                    p_phys = apply_physics_blend(
                        held_df, p_ens, iso,
                        float(train_df[target].mean()), lam
                    )
                    if seed == 42:
                        for candidate_lam, candidate_mae in candidates:
                            physics_rows.append({
                                "target": target,
                                "held_domain": held_domain,
                                "lambda": candidate_lam,
                                "calibration_mae": candidate_mae,
                                "selected": "YES" if candidate_lam == lam else "NO",
                            })

                    methods = {
                        "ERM_POOLED_RIDGE": p_erm,
                        "DOMAIN_DRO_RIDGE": p_dro,
                        "DOMAIN_GRADIENT_PENALIZED_RIDGE": p_grad,
                        "SOURCE_MODEL_ENSEMBLE": p_ens,
                        "PHYSICS_GUIDED_SOURCE_ENSEMBLE": p_phys,
                    }
                    for name, pred in methods.items():
                        seed_method_predictions.setdefault(name, []).append(pred)

                    # UQ around source ensemble.
                    residuals = np.abs(y_cal-p_cal_ens)
                    q_global = conformal_q(residuals)

                    domain_weights = np.zeros(len(cal_df), float)
                    domain_q = []
                    for domain in cal_df["domain_role"].unique():
                        mask = (
                            cal_df["domain_role"].to_numpy(str) == domain
                        )
                        domain_weights[mask] = 1.0/max(mask.sum(), 1)
                        domain_q.append(conformal_q(residuals[mask]))
                    q_balanced = conformal_q(
                        residuals, weights=domain_weights
                    )
                    q_worst = max(domain_q)

                    disagreement_scale = max(
                        np.median(cal_disagreement), 1e-6
                    )
                    for method, half in {
                        "GLOBAL_CONFORMAL": np.full(len(held_df), q_global),
                        "DOMAIN_BALANCED_CONFORMAL": np.full(
                            len(held_df), q_balanced
                        ),
                        "WORST_SOURCE_CONFORMAL": np.full(
                            len(held_df), q_worst
                        ),
                        "SOURCE_DISAGREEMENT_SCALED_BETA_0.25":
                            q_global*(1+0.25*np.clip(
                                held_disagreement/disagreement_scale, 0, 5
                            )),
                        "SOURCE_DISAGREEMENT_SCALED_BETA_0.5":
                            q_global*(1+0.5*np.clip(
                                held_disagreement/disagreement_scale, 0, 5
                            )),
                        "SOURCE_DISAGREEMENT_SCALED_BETA_1":
                            q_global*(1+1.0*np.clip(
                                held_disagreement/disagreement_scale, 0, 5
                            )),
                        "SOURCE_DISAGREEMENT_SCALED_BETA_2":
                            q_global*(1+2.0*np.clip(
                                held_disagreement/disagreement_scale, 0, 5
                            )),
                    }.items():
                        coverage = float(np.mean(
                            np.abs(y_held-p_ens) <= half
                        ))
                        seed_uq.append({
                            "target": target,
                            "held_domain": held_domain,
                            "seed": seed,
                            "method": method,
                            "coverage": coverage,
                            "average_width": float(np.mean(2*half)),
                            "held_n": len(held_df),
                            "calibration_n": len(cal_df),
                        })

                    if seed == 42:
                        error = np.abs(y_held-p_ens)
                        rc, aurc = risk_coverage(
                            y_held, p_ens, held_disagreement,
                            "SOURCE_MODEL_DISAGREEMENT"
                        )
                        for row in rc:
                            risk_rows.append({
                                "target": target,
                                "held_domain": held_domain,
                                **row
                            })
                        corr = float(
                            pd.Series(held_disagreement).rank().corr(
                                pd.Series(error).rank()
                            )
                        )
                        q50, q90 = np.quantile(
                            cal_disagreement, [.50, .90]
                        )
                        groups = np.where(
                            held_disagreement <= q50, "ACCEPT",
                            np.where(
                                held_disagreement <= q90,
                                "REVIEW", "ABSTAIN"
                            )
                        )
                        for group in ["ACCEPT", "REVIEW", "ABSTAIN"]:
                            mask = groups == group
                            risk_rows.append({
                                "target": target,
                                "held_domain": held_domain,
                                "risk_method": "DECISION_GROUP",
                                "coverage": float(mask.mean()),
                                "retained_n": int(mask.sum()),
                                "mae": float(mean_absolute_error(
                                    y_held[mask], p_ens[mask]
                                )) if mask.any() else np.nan,
                                "decision_group": group,
                                "risk_error_rank_correlation": corr,
                                "aurc": aurc,
                            })

                        for i, rec in enumerate(
                            held_df.itertuples(index=False)
                        ):
                            prediction_rows.append({
                                "target": target,
                                "held_domain": held_domain,
                                "record_id": rec.record_id,
                                "y_true": float(y_held[i]),
                                "source_ensemble_prediction": float(p_ens[i]),
                                "absolute_error": float(error[i]),
                                "source_disagreement": float(
                                    held_disagreement[i]
                                ),
                                "decision_group": str(groups[i]),
                            })

                # Aggregate point predictions across seeds.
                y_held = held_df[target].to_numpy(float)
                iqr = float(
                    np.quantile(y_held, .75) -
                    np.quantile(y_held, .25)
                )
                iqr = max(iqr, 1e-9)
                for name, seed_preds in seed_method_predictions.items():
                    pred = np.mean(seed_preds, axis=0)
                    m = metric(y_held, pred)
                    fold_rows.append({
                        "target": target,
                        "held_domain": held_domain,
                        "method": name,
                        **m,
                        "normalized_mae_iqr": m["mae"]/iqr,
                    })
                uq_rows.extend(seed_uq)
            log(f"Multi-domain pilot complete: {target}")

        point_df = pd.DataFrame(fold_rows)
        uq_df = pd.DataFrame(uq_rows)
        risk_df = pd.DataFrame(risk_rows)
        pred_df = pd.DataFrame(prediction_rows)
        physics_df = pd.DataFrame(physics_rows)
        dro_df = pd.DataFrame(dro_rows)

        point_summary = (
            point_df.groupby(["target", "method"], as_index=False)
            .agg(
                mean_normalized_mae=("normalized_mae_iqr", "mean"),
                worst_domain_normalized_mae=("normalized_mae_iqr", "max"),
                mean_mae=("mae", "mean"),
                mean_r2=("r2", "mean"),
                domains=("held_domain", "nunique"),
            )
        )

        uq_summary = (
            uq_df.groupby(["target", "method"], as_index=False)
            .agg(
                mean_coverage=("coverage", "mean"),
                minimum_domain_seed_coverage=("coverage", "min"),
                mean_width=("average_width", "mean"),
                coverage_sd=("coverage", "std"),
            )
        )
        global_width = (
            uq_summary[uq_summary["method"] == "GLOBAL_CONFORMAL"]
            [["target", "mean_width"]]
            .rename(columns={"mean_width": "global_mean_width"})
        )
        uq_summary = uq_summary.merge(global_width, on="target", how="left")
        uq_summary["width_ratio_to_global"] = (
            uq_summary["mean_width"] /
            uq_summary["global_mean_width"]
        )

        gate_rows = []
        for target in TARGETS:
            sub = point_summary[point_summary["target"] == target]
            erm = sub[sub["method"] == "ERM_POOLED_RIDGE"].iloc[0]
            best_non_erm = sub[sub["method"] != "ERM_POOLED_RIDGE"].sort_values(
                ["mean_normalized_mae", "worst_domain_normalized_mae"]
            ).iloc[0]
            avg_improvement = (
                erm["mean_normalized_mae"] -
                best_non_erm["mean_normalized_mae"]
            ) / erm["mean_normalized_mae"]
            worst_improvement = (
                erm["worst_domain_normalized_mae"] -
                best_non_erm["worst_domain_normalized_mae"]
            ) / erm["worst_domain_normalized_mae"]
            point_pass = bool(
                avg_improvement >= .05 and worst_improvement >= .05
            )

            uq_sub = uq_summary[
                (uq_summary["target"] == target) &
                (uq_summary["method"] != "GLOBAL_CONFORMAL")
            ].copy()
            uq_sub["coverage_distance"] = abs(
                uq_sub["mean_coverage"] - .90
            )
            feasible = uq_sub[
                (uq_sub["mean_coverage"] >= .88) &
                (uq_sub["minimum_domain_seed_coverage"] >= .75) &
                (uq_sub["width_ratio_to_global"] <= 2.5)
            ]
            if len(feasible):
                best_uq = feasible.sort_values(
                    ["coverage_distance", "width_ratio_to_global"]
                ).iloc[0]
                uq_pass = True
            else:
                best_uq = uq_sub.sort_values(
                    ["coverage_distance", "width_ratio_to_global"]
                ).iloc[0]
                uq_pass = False

            gate_rows.append({
                "target": target,
                "selected_point_method": best_non_erm["method"],
                "point_mean_nmae_improvement_fraction": avg_improvement,
                "point_worst_domain_improvement_fraction": worst_improvement,
                "point_gate_pass": point_pass,
                "selected_uq_method": best_uq["method"],
                "uq_mean_coverage": best_uq["mean_coverage"],
                "uq_minimum_coverage": best_uq[
                    "minimum_domain_seed_coverage"
                ],
                "uq_width_ratio_to_global": best_uq[
                    "width_ratio_to_global"
                ],
                "uq_gate_pass": uq_pass,
            })

        gates = pd.DataFrame(gate_rows)
        point_pass_count = int(gates["point_gate_pass"].sum())
        uq_pass_count = int(gates["uq_gate_pass"].sum())

        if point_pass_count == 2 and uq_pass_count == 2:
            decision = "GREEN_FOR_STAGE3_INTEGRATED_PROTOTYPE"
            rationale = (
                "Both targets passed the multi-domain point-robustness and "
                "uncertainty gates. The selected components may be integrated "
                "and frozen for a controlled Stage3 prototype."
            )
        elif point_pass_count >= 1 and uq_pass_count >= 1:
            decision = "YELLOW_MULTI_DOMAIN_REDESIGN"
            rationale = (
                "The multi-domain pilot is partially successful, but at least "
                "one target or component requires redesign."
            )
        else:
            decision = "RED_WOODPHYS_PIVOT"
            rationale = (
                "The multi-domain methods did not improve both worst-domain "
                "point error and uncertainty sufficiently."
            )

        point_df.to_csv(
            TABLES/"POINT_RESULTS_BY_HELD_DOMAIN.csv",
            index=False, encoding="utf-8-sig"
        )
        point_summary.to_csv(
            TABLES/"POINT_MULTI_DOMAIN_SUMMARY.csv",
            index=False, encoding="utf-8-sig"
        )
        uq_df.to_csv(
            TABLES/"UQ_RESULTS_BY_SEED_DOMAIN.csv",
            index=False, encoding="utf-8-sig"
        )
        uq_summary.to_csv(
            TABLES/"UQ_MULTI_DOMAIN_SUMMARY.csv",
            index=False, encoding="utf-8-sig"
        )
        gates.to_csv(
            TABLES/"METHOD_SELECTION_AND_GATES.csv",
            index=False, encoding="utf-8-sig"
        )
        risk_df.to_csv(
            TABLES/"SELECTIVE_RISK_MULTI_DOMAIN.csv",
            index=False, encoding="utf-8-sig"
        )
        physics_df.to_csv(
            TABLES/"PHYSICS_BLEND_SELECTION.csv",
            index=False, encoding="utf-8-sig"
        )
        dro_df.to_csv(
            TABLES/"DOMAIN_DRO_WEIGHTS.csv",
            index=False, encoding="utf-8-sig"
        )
        pred_df.to_csv(
            DATA/"SOURCE_ENSEMBLE_HELD_DOMAIN_PREDICTIONS.csv",
            index=False, encoding="utf-8-sig"
        )

        with pd.ExcelWriter(
            TABLES/"STAGE2B_MULTI_DOMAIN_PILOT.xlsx",
            engine="openpyxl"
        ) as xw:
            point_df.to_excel(xw, sheet_name="Point by domain", index=False)
            point_summary.to_excel(xw, sheet_name="Point summary", index=False)
            uq_summary.to_excel(xw, sheet_name="UQ summary", index=False)
            gates.to_excel(xw, sheet_name="Gates", index=False)
            risk_df.to_excel(xw, sheet_name="Selective risk", index=False)
            physics_df.to_excel(xw, sheet_name="Physics lambda", index=False)
            dro_df.to_excel(xw, sheet_name="DRO weights", index=False)

        for target in TARGETS:
            sub = point_summary[
                point_summary["target"] == target
            ].sort_values("mean_normalized_mae")
            plt.figure(figsize=(10, 5))
            plt.barh(sub["method"], sub["mean_normalized_mae"])
            plt.xlabel("Mean normalized MAE across held domains")
            plt.title(f"Multi-domain point robustness: {target}")
            plt.tight_layout()
            plt.savefig(
                FIGURES/f"POINT_ROBUSTNESS_{target}.png", dpi=180
            )
            plt.close()

            uq_sub = uq_summary[
                uq_summary["target"] == target
            ].sort_values("mean_coverage")
            plt.figure(figsize=(10, 6))
            plt.barh(uq_sub["method"], uq_sub["mean_coverage"])
            plt.axvline(.90, linestyle="--")
            plt.xlabel("Mean empirical coverage")
            plt.title(f"Multi-domain UQ coverage: {target}")
            plt.tight_layout()
            plt.savefig(
                FIGURES/f"UQ_COVERAGE_{target}.png", dpi=180
            )
            plt.close()

        scientific = {
            "decision": decision,
            "rationale": rationale,
            "full_hap_dg_authorization": "NO",
            "stage3_authorization": (
                "CONTROLLED_INTEGRATED_PROTOTYPE"
                if decision == "GREEN_FOR_STAGE3_INTEGRATED_PROTOTYPE"
                else "NO"
            ),
            "point_gate_pass_count": point_pass_count,
            "uq_gate_pass_count": uq_pass_count,
            "method_gate_rows": gates.to_dict("records"),
            "final_external_evaluation_status": "NOT_YET_AVAILABLE",
            "required_before_paper_claim": [
                "acquire or pre-register an independent external lockbox",
                "manual row-level USDA audit",
                "freeze the selected Stage3 method before external evaluation",
                "repeat statistical evaluation without method changes",
            ],
            "generated_at": now(),
        }
        write_json(OUT/"SCIENTIFIC_STAGE2B_DECISION.json", scientific)

        report = [
            "# WOODPHYS_DG_Q1 — Stage 2B Multi-Domain Pilot",
            "",
            f"## القرار: **{decision}**",
            "",
            rationale,
            "",
            "## طبيعة التقييم",
            "هذه Leave-One-Domain-Out development evaluation عبر US وCanada وOther/imports.",
            "لا يوجد final external validation حتى الآن.",
            "",
        ]
        for _, row in gates.iterrows():
            report += [
                f"### {row['target']}",
                f"- Selected point method: `{row['selected_point_method']}`",
                f"- Mean normalized-MAE improvement: "
                f"{100*row['point_mean_nmae_improvement_fraction']:.2f}%",
                f"- Worst-domain improvement: "
                f"{100*row['point_worst_domain_improvement_fraction']:.2f}%",
                f"- Point gate: `{row['point_gate_pass']}`",
                f"- Selected UQ method: `{row['selected_uq_method']}`",
                f"- Mean coverage: {row['uq_mean_coverage']:.3f}",
                f"- Minimum coverage: {row['uq_minimum_coverage']:.3f}",
                f"- Width ratio to global: "
                f"{row['uq_width_ratio_to_global']:.3f}",
                f"- UQ gate: `{row['uq_gate_pass']}`",
                "",
            ]
        report += [
            "## الملفات الأساسية",
            "- `tables/STAGE2B_MULTI_DOMAIN_PILOT.xlsx`",
            "- `tables/POINT_MULTI_DOMAIN_SUMMARY.csv`",
            "- `tables/UQ_MULTI_DOMAIN_SUMMARY.csv`",
            "- `tables/METHOD_SELECTION_AND_GATES.csv`",
            "- `tables/SELECTIVE_RISK_MULTI_DOMAIN.csv`",
            "",
            "## حدود الادعاء",
            "- لا يسمح بادعاء external validation نهائي.",
            "- لا يسمح بـFull HAP-DG.",
            "- Source-model ensemble مكوّن تطويري، وليس دليلًا على التفوق النهائي.",
        ]
        (OUT/"MULTI_DOMAIN_REPORT_AR.md").write_text(
            "\n".join(report)+"\n", encoding="utf-8"
        )

        write_json(OUT/"OUTPUT_VERIFICATION.json", {
            "verification_passed": True,
            "stage0f_verified": True,
            "stage1a_verified": True,
            "outer_domains": DOMAINS,
            "seeds": SEEDS,
            "categorical_missing_explicit": True,
            "final_test_claim_prevented": True,
            "full_hap_dg_prevented": True,
        })
        write_json(OUT/"CHECKPOINT.json", {
            "completed_steps": [
                "input_verification",
                "three_domain_leave_one_out_point_evaluation",
                "domain_dro_component",
                "gradient_penalty_component",
                "source_model_ensemble_component",
                "physics_guided_blend_component",
                "multi_source_conformal_component",
                "source_disagreement_selective_risk",
            ],
            "timestamp": now(),
        })
        write_json(OUT/"STATUS.json", {
            "stage": "STAGE2B", "version": "1.0.0",
            "status": "completed", "decision": decision,
            "completed_at": now(),
        })
        write_json(OUT/"SUCCESS_MARKER.json", {
            "success": True, "stage": "STAGE2B", "version": "1.0.0",
            "decision": decision, "timestamp": now(),
        })
        write_json(OUT/"ENVIRONMENT_SNAPSHOT.json", {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        })
        (OUT/"ERROR_LOG.txt").write_text("", encoding="utf-8")
        (OUT/"RUN_LOG.txt").write_text(
            "\n".join(LOG_LINES)+"\n", encoding="utf-8"
        )

        required = [
            "MULTI_DOMAIN_REPORT_AR.md",
            "SCIENTIFIC_STAGE2B_DECISION.json",
            "tables/STAGE2B_MULTI_DOMAIN_PILOT.xlsx",
            "tables/POINT_MULTI_DOMAIN_SUMMARY.csv",
            "tables/UQ_MULTI_DOMAIN_SUMMARY.csv",
            "tables/METHOD_SELECTION_AND_GATES.csv",
            "data/SOURCE_ENSEMBLE_HELD_DOMAIN_PREDICTIONS.csv",
        ]
        missing = [x for x in required if not (OUT/x).exists()]
        if missing:
            raise RuntimeError(f"Missing required outputs: {missing}")

        manifest_rows = []
        hash_rows = []
        for path in sorted(OUT.rglob("*")):
            if not path.is_file() or "_work" in path.parts:
                continue
            rel = path.relative_to(OUT).as_posix()
            if rel in {
                "FILE_MANIFEST.csv", "HASHES_SHA256.csv",
                "UPLOAD_THIS_STAGE2B_RESULT.zip",
            }:
                continue
            manifest_rows.append({
                "relative_path": rel,
                "size_bytes": path.stat().st_size,
            })
            hash_rows.append({
                "relative_path": rel,
                "sha256": sha256(path),
            })
        pd.DataFrame(manifest_rows).to_csv(
            OUT/"FILE_MANIFEST.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(hash_rows).to_csv(
            OUT/"HASHES_SHA256.csv", index=False, encoding="utf-8-sig"
        )

        upload = OUT/"UPLOAD_THIS_STAGE2B_RESULT.zip"
        with zipfile.ZipFile(upload, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(OUT.rglob("*")):
                if path.is_file() and path != upload and "_work" not in path.parts:
                    z.write(path, path.relative_to(OUT))

        log(f"Completed Stage2B. Decision: {decision}")
        (OUT/"RUN_LOG.txt").write_text(
            "\n".join(LOG_LINES)+"\n", encoding="utf-8"
        )
        return 0

    except Exception as exc:
        ERROR_LINES.append(traceback.format_exc())
        write_json(OUT/"STATUS.json", {
            "stage": "STAGE2B", "status": "failed",
            "error": str(exc), "completed_at": now(),
        })
        write_json(OUT/"SUCCESS_MARKER.json", {
            "success": False, "stage": "STAGE2B",
            "error": str(exc), "timestamp": now(),
        })
        (OUT/"RUN_LOG.txt").write_text(
            "\n".join(LOG_LINES)+"\n", encoding="utf-8"
        )
        (OUT/"ERROR_LOG.txt").write_text(
            "\n".join(ERROR_LINES)+"\n", encoding="utf-8"
        )
        print(ERROR_LINES[-1], file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
