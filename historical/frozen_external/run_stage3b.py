
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

from scipy.stats import wilcoxon, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT0F = ROOT / "input" / "STAGE0F_VERIFIED_INPUT.zip"
INPUTLOCK = ROOT / "input" / "STAGE3A_FIX2_GREEN_LOCKBOX_INPUT.zip"
PROTOCOL = ROOT / "protocol" / "EXTERNAL_TRANSPORT_PROTOCOL.json"

OUT = ROOT / "stage3b_output"
WORK = OUT / "_work"
TABLES = OUT / "tables"
DATA = OUT / "data"
FIGURES = OUT / "figures"
AUDIT = OUT / "audit"

DOMAINS = ["DEVELOPMENT_US", "CALIBRATION_CANADA", "EXTERNAL_OTHER_IMPORTS"]
TARGETS = ["moe_mpa", "mor_mpa"]
PARAMS = {
    "moe_mpa": {"temperature": 3.0, "beta": 0.75},
    "mor_mpa": {"temperature": 0.2, "beta": 1.0},
}
SEEDS = [13, 42, 97, 131, 2026]
ALPHA = 0.10
BOOT = 2000

FEATURES = [
    "specific_gravity", "moisture_percent",
    "moisture_state", "wood_class", "genus", "family"
]
NUMS = ["specific_gravity", "moisture_percent"]
CATS = ["moisture_state", "wood_class", "genus", "family"]

LOG_LINES = []
ERROR_LINES = []

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

def metric(y, pred):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    iqr = max(float(np.quantile(y, .75)-np.quantile(y, .25)), 1e-9)
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred)**.5),
        "r2": float(r2_score(y, pred)),
        "normalized_mae_iqr": float(mean_absolute_error(y, pred)/iqr)
    }

def bootstrap_metric_ci(y, pred, seed=2026):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    rng = np.random.default_rng(seed)
    vals = {"mae": [], "rmse": [], "r2": []}
    for _ in range(BOOT):
        idx = rng.integers(0, len(y), len(y))
        yy, pp = y[idx], pred[idx]
        vals["mae"].append(mean_absolute_error(yy, pp))
        vals["rmse"].append(mean_squared_error(yy, pp)**.5)
        try:
            vals["r2"].append(r2_score(yy, pp))
        except Exception:
            vals["r2"].append(np.nan)
    out = {}
    for key, arr in vals.items():
        arr = np.asarray(arr, float)
        out[f"{key}_ci_low"] = float(np.nanquantile(arr, .025))
        out[f"{key}_ci_high"] = float(np.nanquantile(arr, .975))
    return out

def conformal_q(residuals):
    r = np.sort(np.asarray(residuals, float))
    k = min(len(r)-1, max(0, int(math.ceil((len(r)+1)*(1-ALPHA)))-1))
    return float(r[k])

def verify_inputs():
    s0 = WORK / "stage0f"
    sl = WORK / "lockbox"
    s0.mkdir(parents=True, exist_ok=True)
    sl.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(INPUT0F) as z:
        z.extractall(s0)
    with zipfile.ZipFile(INPUTLOCK) as z:
        z.extractall(sl)

    s0_success = json.loads((s0/"SUCCESS_MARKER.json").read_text(encoding="utf-8"))
    s0_verify = json.loads((s0/"OUTPUT_VERIFICATION.json").read_text(encoding="utf-8"))
    lock_success = json.loads((sl/"SUCCESS_MARKER.json").read_text(encoding="utf-8"))
    lock_verify = json.loads((sl/"OUTPUT_VERIFICATION.json").read_text(encoding="utf-8"))
    lock_decision = json.loads(
        (sl/"SCIENTIFIC_STAGE3A_FIX2_DECISION.json").read_text(encoding="utf-8")
    )

    if not s0_success.get("success") or not s0_verify.get("verification_passed"):
        raise RuntimeError("Stage0F input verification failed")
    if not lock_success.get("success") or not lock_verify.get("verification_passed"):
        raise RuntimeError("Stage3A FIX2 input verification failed")
    if lock_decision.get("decision") != "GREEN_PINUS_LOCKBOX_REGISTERED_AND_QUARANTINED":
        raise RuntimeError("The external lockbox is not green")
    if lock_verify.get("external_predictions_computed"):
        raise RuntimeError("Lockbox was already used for prediction before Stage3B")
    if lock_verify.get("external_metrics_computed"):
        raise RuntimeError("Lockbox metrics were computed before Stage3B")

    for root in [s0, sl]:
        hash_file = root / "HASHES_SHA256.csv"
        rows = pd.read_csv(hash_file)
        mismatches = []
        for _, row in rows.iterrows():
            path = root / row["relative_path"]
            actual = sha256(path) if path.exists() else "MISSING"
            if actual != row["sha256"]:
                mismatches.append(row["relative_path"])
        if mismatches:
            raise RuntimeError(f"Hash mismatch in {root.name}: {mismatches[:5]}")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    frozen = protocol["frozen_parameters"]
    if frozen["moe_mpa"] != {
        "applicability_temperature": 3.0, "uq_beta": 0.75
    }:
        raise RuntimeError("MOE frozen parameter mismatch")
    if frozen["mor_mpa"] != {
        "applicability_temperature": 0.2, "uq_beta": 1.0
    }:
        raise RuntimeError("MOR frozen parameter mismatch")

    board = sl / "quarantine" / "Board_Data.csv"
    if not board.exists():
        raise RuntimeError("Board_Data.csv is missing from quarantine")
    expected_hash = "705c84725af8fe252624321d51a348f04f8f2d42e0322f32dc07d2156c22e0cd"
    if sha256(board) != expected_hash:
        raise RuntimeError("Board_Data.csv hash differs from the registered lockbox")

    log("Verified Stage0F, Stage3A FIX2, method freeze, protocol, and lockbox hash")
    return s0, sl, board

def external_frame(board_path):
    raw = pd.read_csv(board_path, encoding="utf-8-sig")
    required = ["BoardID", "Spacing", "MOE", "MOR", "Density"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"External schema missing: {missing}")

    mapped = pd.DataFrame({
        "record_id": raw["BoardID"].astype(str),
        "spacing": raw["Spacing"],
        "tree": raw["Tree"].astype(str),
        "specific_gravity": pd.to_numeric(raw["Density"], errors="coerce")/1000.0,
        "moisture_percent": np.nan,
        "moisture_state": "__MISSING__",
        "wood_class": "softwood",
        "genus": "Pinus",
        "family": "Pinaceae",
        "moe_mpa": pd.to_numeric(raw["MOE"], errors="coerce"),
        "mor_mpa": pd.to_numeric(raw["MOR"], errors="coerce"),
    })
    if mapped["record_id"].duplicated().any():
        raise RuntimeError("Duplicate external BoardID detected")
    return raw, mapped

def split_source(source_df, seed):
    train_parts, cal_parts = [], []
    for _, group in source_df.groupby("domain_role"):
        groups = group["species_group_key"].fillna(
            group["species_label_raw"]
        ).astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=.20, random_state=seed)
        tr, ca = next(splitter.split(group, groups=groups))
        train_parts.append(group.iloc[tr])
        cal_parts.append(group.iloc[ca])
    return pd.concat(train_parts, ignore_index=True), pd.concat(cal_parts, ignore_index=True)

def fit_source_models(x, y, domains):
    models, centroids = {}, {}
    for domain in sorted(np.unique(domains)):
        mask = domains == domain
        model = Ridge(alpha=1.0)
        model.fit(x[mask], y[mask])
        models[domain] = model
        centroids[domain] = x[mask].mean(axis=0)
    return models, centroids

def source_matrix(models, x):
    names = sorted(models)
    return names, np.vstack([models[name].predict(x) for name in names])

def applicability_predict(models, centroids, x, temperature):
    names, matrix = source_matrix(models, x)
    distances = np.vstack([
        np.linalg.norm(x-centroids[name], axis=1) for name in names
    ])
    weights = np.exp(-distances/temperature)
    weights /= weights.sum(axis=0, keepdims=True)
    pred = (weights*matrix).sum(axis=0)
    disagreement = matrix.std(axis=0)
    return pred, disagreement, weights, distances, matrix

def development_calibration_pool(usda, target, temperature):
    residuals = []
    disagreements = []
    for held_domain in DOMAINS:
        source_df = usda[usda["domain_role"] != held_domain].copy()
        for seed in SEEDS:
            train_df, cal_df = split_source(source_df, seed)
            pre = preprocessor()
            xtr = np.asarray(pre.fit_transform(train_df[FEATURES]), float)
            xcal = np.asarray(pre.transform(cal_df[FEATURES]), float)
            models, centroids = fit_source_models(
                xtr, train_df[target].to_numpy(float),
                train_df["domain_role"].to_numpy(str)
            )
            pred, disagreement, _, _, _ = applicability_predict(
                models, centroids, xcal, temperature
            )
            residuals.extend(
                np.abs(cal_df[target].to_numpy(float)-pred).tolist()
            )
            disagreements.extend(disagreement.tolist())
    return np.asarray(residuals, float), np.asarray(disagreements, float)

def holm_adjust(p_values):
    p_values = np.asarray(p_values, float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), float)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        value = min(1.0, (m-rank)*p_values[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted

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
        [row["mae"] for row in rows],
        [row["coverage"] for row in rows]
    ))
    return rows, aurc

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for directory in [OUT, WORK, TABLES, DATA, FIGURES, AUDIT]:
        directory.mkdir(parents=True, exist_ok=True)

    write_json(OUT/"STATUS.json", {
        "stage": "STAGE3B", "version": "1.0.0",
        "status": "running", "started_at": now()
    })

    try:
        stage0f, lockbox_root, board_path = verify_inputs()

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

        raw_board, external = external_frame(board_path)
        external_mapping = {
            "source_rows": int(len(raw_board)),
            "unique_board_ids": int(external["record_id"].nunique()),
            "moe_missing": int(external["moe_mpa"].isna().sum()),
            "mor_missing": int(external["mor_mpa"].isna().sum()),
            "density_missing": int(external["specific_gravity"].isna().sum()),
            "feature_mapping": json.loads(PROTOCOL.read_text(encoding="utf-8"))[
                "feature_mapping"
            ],
            "measurement_compatibility": (
                "OUT_OF_REGIME: structural lumber four-point bending versus "
                "USDA clear-wood species/group means"
            )
        }
        write_json(AUDIT/"EXTERNAL_MAPPING_AUDIT.json", external_mapping)

        metric_rows = []
        interval_rows = []
        prediction_rows = []
        comparison_rows = []
        risk_rows = []
        spacing_rows = []
        weight_rows = []

        target_gate_rows = []

        for target in TARGETS:
            log(f"Frozen external evaluation start: {target}")
            params = PARAMS[target]
            complete = external[external[target].notna()].copy()
            y = complete[target].to_numpy(float)

            # Final frozen fit: one source-specific model per all three USDA domains.
            pre = preprocessor()
            x_dev = np.asarray(pre.fit_transform(usda[FEATURES]), float)
            x_ext = np.asarray(pre.transform(complete[FEATURES]), float)
            y_dev = usda[target].to_numpy(float)
            dev_domains = usda["domain_role"].to_numpy(str)

            erm = Ridge(alpha=1.0).fit(x_dev, y_dev)
            pred_erm = erm.predict(x_ext)

            models, centroids = fit_source_models(
                x_dev, y_dev, dev_domains
            )
            _, matrix = source_matrix(models, x_ext)
            pred_equal = matrix.mean(axis=0)
            pred_frozen, disagreement, weights, distances, source_matrix_values = (
                applicability_predict(
                    models, centroids, x_ext, params["temperature"]
                )
            )

            method_preds = {
                "ERM_POOLED_RIDGE": pred_erm,
                "EQUAL_SOURCE_ENSEMBLE": pred_equal,
                "FROZEN_APPLICABILITY_WEIGHTED_ENSEMBLE": pred_frozen,
            }

            for method, pred in method_preds.items():
                row = {
                    "target": target,
                    "method": method,
                    **metric(y, pred),
                    **bootstrap_metric_ci(y, pred)
                }
                metric_rows.append(row)

            # Predefined paired comparisons.
            absolute_errors = {
                method: np.abs(y-pred)
                for method, pred in method_preds.items()
            }
            comparisons = [
                ("FROZEN_APPLICABILITY_WEIGHTED_ENSEMBLE", "ERM_POOLED_RIDGE"),
                ("FROZEN_APPLICABILITY_WEIGHTED_ENSEMBLE", "EQUAL_SOURCE_ENSEMBLE"),
            ]
            local_p = []
            local_rows = []
            for a, b in comparisons:
                diff = absolute_errors[a]-absolute_errors[b]
                try:
                    stat, p_value = wilcoxon(
                        absolute_errors[a], absolute_errors[b],
                        alternative="two-sided", zero_method="wilcox"
                    )
                except Exception:
                    stat, p_value = np.nan, 1.0
                local_p.append(p_value)
                local_rows.append({
                    "target": target,
                    "method_a": a,
                    "method_b": b,
                    "mean_abs_error_a": float(absolute_errors[a].mean()),
                    "mean_abs_error_b": float(absolute_errors[b].mean()),
                    "mean_paired_difference_a_minus_b": float(diff.mean()),
                    "wilcoxon_statistic": float(stat) if np.isfinite(stat) else np.nan,
                    "p_raw": float(p_value)
                })
            adjusted = holm_adjust(local_p)
            for row, adj in zip(local_rows, adjusted):
                row["p_holm"] = float(adj)
                row["significant_0_05"] = bool(adj < .05)
                comparison_rows.append(row)

            # Frozen UQ calibrated only on development-domain residuals.
            cal_residuals, cal_disagreement = development_calibration_pool(
                usda, target, params["temperature"]
            )
            q_global = conformal_q(cal_residuals)
            disagreement_scale = max(float(np.median(cal_disagreement)), 1e-6)
            global_half = np.full(len(y), q_global)
            frozen_half = q_global * (
                1 + params["beta"] *
                np.clip(disagreement/disagreement_scale, 0, 5)
            )

            for method, half in [
                ("DEVELOPMENT_GLOBAL_CONFORMAL", global_half),
                ("FROZEN_DISAGREEMENT_SCALED_CONFORMAL", frozen_half),
            ]:
                covered = np.abs(y-pred_frozen) <= half
                interval_rows.append({
                    "target": target,
                    "method": method,
                    "coverage": float(covered.mean()),
                    "average_width": float(np.mean(2*half)),
                    "median_width": float(np.median(2*half)),
                    "width_ratio_to_global": float(
                        np.mean(half)/max(q_global, 1e-9)
                    ),
                    "external_n": int(len(y)),
                    "development_calibration_residual_n": int(len(cal_residuals)),
                    "external_labels_used_for_calibration": False,
                })

            # Selective risk.
            corr_result = spearmanr(disagreement, np.abs(y-pred_frozen))
            risk_corr = float(corr_result.statistic)
            rc, aurc = risk_coverage(y, pred_frozen, disagreement)
            for row in rc:
                risk_rows.append({
                    "target": target,
                    "risk_method": "SOURCE_MODEL_DISAGREEMENT",
                    **row,
                    "aurc": aurc,
                    "risk_error_spearman": risk_corr
                })

            q50, q90 = np.quantile(cal_disagreement, [.50, .90])
            decisions = np.where(
                disagreement <= q50, "ACCEPT",
                np.where(disagreement <= q90, "REVIEW", "ABSTAIN")
            )
            decision_mae = {}
            for group in ["ACCEPT", "REVIEW", "ABSTAIN"]:
                mask = decisions == group
                value = (
                    float(mean_absolute_error(y[mask], pred_frozen[mask]))
                    if mask.any() else np.nan
                )
                decision_mae[group] = value
                risk_rows.append({
                    "target": target,
                    "risk_method": "DECISION_GROUP",
                    "coverage": float(mask.mean()),
                    "retained_n": int(mask.sum()),
                    "mae": value,
                    "decision_group": group,
                    "aurc": aurc,
                    "risk_error_spearman": risk_corr
                })

            # Spacing subgroups.
            for spacing, group in complete.assign(
                _y=y, _pred=pred_frozen, _risk=disagreement
            ).groupby("spacing"):
                spacing_rows.append({
                    "target": target,
                    "spacing": spacing,
                    "n": int(len(group)),
                    "mae": float(mean_absolute_error(group["_y"], group["_pred"])),
                    "rmse": float(mean_squared_error(group["_y"], group["_pred"])**.5),
                    "mean_source_disagreement": float(group["_risk"].mean())
                })

            source_names = sorted(models)
            for i, rec in enumerate(complete.itertuples(index=False)):
                row = {
                    "target": target,
                    "record_id": rec.record_id,
                    "spacing": rec.spacing,
                    "tree": rec.tree,
                    "y_true": float(y[i]),
                    "pred_erm": float(pred_erm[i]),
                    "pred_equal_ensemble": float(pred_equal[i]),
                    "pred_frozen": float(pred_frozen[i]),
                    "absolute_error_frozen": float(abs(y[i]-pred_frozen[i])),
                    "source_disagreement": float(disagreement[i]),
                    "decision_group": str(decisions[i]),
                    "global_lower": float(pred_frozen[i]-global_half[i]),
                    "global_upper": float(pred_frozen[i]+global_half[i]),
                    "frozen_lower": float(pred_frozen[i]-frozen_half[i]),
                    "frozen_upper": float(pred_frozen[i]+frozen_half[i]),
                }
                for j, source in enumerate(source_names):
                    row[f"weight_{source}"] = float(weights[j, i])
                    row[f"prediction_{source}"] = float(source_matrix_values[j, i])
                    weight_rows.append({
                        "target": target,
                        "record_id": rec.record_id,
                        "source_domain": source,
                        "weight": float(weights[j, i]),
                        "distance": float(distances[j, i])
                    })
                prediction_rows.append(row)

            frozen_mae = mean_absolute_error(y, pred_frozen)
            erm_mae = mean_absolute_error(y, pred_erm)
            equal_mae = mean_absolute_error(y, pred_equal)
            point_improvement_erm = (erm_mae-frozen_mae)/max(erm_mae, 1e-9)
            equal_noninferiority = frozen_mae <= equal_mae*1.02
            point_pass = bool(
                point_improvement_erm >= .05 and equal_noninferiority
            )

            frozen_interval = next(
                row for row in interval_rows
                if row["target"] == target and
                row["method"] == "FROZEN_DISAGREEMENT_SCALED_CONFORMAL"
            )
            uq_pass = bool(
                frozen_interval["coverage"] >= .85 and
                frozen_interval["width_ratio_to_global"] <= 2.5
            )
            accept_mae = decision_mae.get("ACCEPT", np.nan)
            selective_pass = bool(
                risk_corr >= .15 and
                np.isfinite(accept_mae) and accept_mae < frozen_mae
            )

            target_gate_rows.append({
                "target": target,
                "point_improvement_vs_erm_fraction": point_improvement_erm,
                "frozen_vs_equal_ensemble_mae_ratio": frozen_mae/max(equal_mae, 1e-9),
                "point_support_gate": point_pass,
                "frozen_interval_coverage": frozen_interval["coverage"],
                "frozen_interval_width_ratio_to_global":
                    frozen_interval["width_ratio_to_global"],
                "uq_support_gate": uq_pass,
                "risk_error_spearman": risk_corr,
                "accept_mae": accept_mae,
                "full_frozen_mae": frozen_mae,
                "selective_support_gate": selective_pass
            })
            log(
                f"External evaluation complete: {target}; "
                f"point={point_pass}, uq={uq_pass}, selective={selective_pass}"
            )

        metrics_df = pd.DataFrame(metric_rows)
        intervals_df = pd.DataFrame(interval_rows)
        comparisons_df = pd.DataFrame(comparison_rows)
        risk_df = pd.DataFrame(risk_rows)
        spacing_df = pd.DataFrame(spacing_rows)
        predictions_df = pd.DataFrame(prediction_rows)
        weights_df = pd.DataFrame(weight_rows)
        gates_df = pd.DataFrame(target_gate_rows)

        support_counts = {
            "point": int(gates_df["point_support_gate"].sum()),
            "uq": int(gates_df["uq_support_gate"].sum()),
            "selective": int(gates_df["selective_support_gate"].sum())
        }
        if support_counts["point"] >= 1 and support_counts["uq"] == 2:
            decision = "GREEN_TRANSPORTABILITY_SUPPORT_PENDING_SAME_REGIME_LOCKBOX"
            rationale = (
                "The frozen method received point support for at least one target and "
                "uncertainty support for both targets on the independent structural-lumber "
                "lockbox. This supports transportability, but a same-regime clear-wood "
                "external lockbox is still required for the primary claim."
            )
        elif support_counts["point"] + support_counts["uq"] + support_counts["selective"] >= 2:
            decision = "YELLOW_MIXED_TRANSPORTABILITY_EVIDENCE"
            rationale = (
                "The independent lockbox provides partial support, but point accuracy, "
                "uncertainty, or selective risk is not consistently successful."
            )
        else:
            decision = "RED_TRANSPORTABILITY_FAILURE"
            rationale = (
                "The frozen method did not transport successfully from USDA clear-wood "
                "domains to Pinus patula structural lumber."
            )

        metrics_df.to_csv(
            TABLES/"EXTERNAL_POINT_METRICS.csv", index=False, encoding="utf-8-sig"
        )
        intervals_df.to_csv(
            TABLES/"EXTERNAL_INTERVAL_METRICS.csv", index=False, encoding="utf-8-sig"
        )
        comparisons_df.to_csv(
            TABLES/"WILCOXON_HOLM_COMPARISONS.csv", index=False, encoding="utf-8-sig"
        )
        risk_df.to_csv(
            TABLES/"EXTERNAL_SELECTIVE_RISK.csv", index=False, encoding="utf-8-sig"
        )
        spacing_df.to_csv(
            TABLES/"EXTERNAL_SPACING_SUBGROUPS.csv", index=False, encoding="utf-8-sig"
        )
        gates_df.to_csv(
            TABLES/"EXTERNAL_SUPPORT_GATES.csv", index=False, encoding="utf-8-sig"
        )
        predictions_df.to_csv(
            DATA/"EXTERNAL_FROZEN_PREDICTIONS.csv", index=False, encoding="utf-8-sig"
        )
        weights_df.to_csv(
            DATA/"EXTERNAL_SOURCE_WEIGHTS.csv", index=False, encoding="utf-8-sig"
        )

        with pd.ExcelWriter(
            TABLES/"STAGE3B_EXTERNAL_RESULTS.xlsx", engine="openpyxl"
        ) as xw:
            metrics_df.to_excel(xw, sheet_name="Point metrics", index=False)
            intervals_df.to_excel(xw, sheet_name="Intervals", index=False)
            comparisons_df.to_excel(xw, sheet_name="Wilcoxon Holm", index=False)
            gates_df.to_excel(xw, sheet_name="Support gates", index=False)
            spacing_df.to_excel(xw, sheet_name="Spacing subgroups", index=False)
            risk_df.to_excel(xw, sheet_name="Selective risk", index=False)

        # Figures.
        for target in TARGETS:
            sub = metrics_df[metrics_df["target"] == target].sort_values("mae")
            plt.figure(figsize=(9, 5))
            plt.barh(sub["method"], sub["mae"])
            plt.xlabel("External MAE")
            plt.title(f"Frozen external transportability: {target}")
            plt.tight_layout()
            plt.savefig(FIGURES/f"EXTERNAL_MAE_{target}.png", dpi=180)
            plt.close()

            sub_risk = risk_df[
                (risk_df["target"] == target) &
                (risk_df["risk_method"] == "SOURCE_MODEL_DISAGREEMENT")
            ]
            plt.figure(figsize=(8, 5))
            plt.plot(sub_risk["coverage"], sub_risk["mae"], marker="o")
            plt.xlabel("Retained fraction")
            plt.ylabel("External MAE")
            plt.title(f"External risk–coverage: {target}")
            plt.tight_layout()
            plt.savefig(FIGURES/f"EXTERNAL_RISK_COVERAGE_{target}.png", dpi=180)
            plt.close()

        scientific = {
            "decision": decision,
            "rationale": rationale,
            "evaluation_role": (
                "INDEPENDENT_OUT_OF_REGIME_TRANSPORTABILITY_STRESS_TEST"
            ),
            "same_regime_external_validation": False,
            "method_parameters_changed_after_lockbox": False,
            "external_labels_used_for_training": False,
            "external_labels_used_for_calibration": False,
            "external_rows_total": int(len(external)),
            "target_support_gates": gates_df.to_dict("records"),
            "support_counts": support_counts,
            "required_next_steps": [
                "retain all Stage3B results without method modification",
                "search for an independent clear-wood same-regime MOE/MOR lockbox",
                "audit USDA extraction rows and reuse permissions",
                "use Stage3B as transportability evidence, not primary validation"
            ],
            "generated_at": now()
        }
        write_json(OUT/"SCIENTIFIC_STAGE3B_DECISION.json", scientific)

        report = [
            "# WOODPHYS_DG_Q1 — Stage 3B Frozen External Test",
            "",
            f"## القرار: **{decision}**",
            "",
            rationale,
            "",
            "## نوع الاختبار",
            "**Independent out-of-regime transportability stress test**",
            "",
            "USDA تمثل clear-wood species/group means، بينما Pinus patula تمثل",
            "structural lumber boards. لذلك لا يجوز وصف النتيجة بأنها same-regime",
            "external validation مهما كانت النتيجة جيدة.",
            "",
            "## سلامة التجميد",
            "- Frozen temperatures changed: NO",
            "- Frozen UQ betas changed: NO",
            "- External labels used for training: NO",
            "- External labels used for calibration: NO",
            f"- External board rows: {len(external)}",
            "",
        ]
        for _, row in gates_df.iterrows():
            report += [
                f"### {row['target']}",
                f"- Improvement vs ERM: "
                f"{100*row['point_improvement_vs_erm_fraction']:.2f}%",
                f"- Frozen/equal-ensemble MAE ratio: "
                f"{row['frozen_vs_equal_ensemble_mae_ratio']:.3f}",
                f"- Point support: `{row['point_support_gate']}`",
                f"- Frozen interval coverage: "
                f"{row['frozen_interval_coverage']:.3f}",
                f"- Width ratio to global: "
                f"{row['frozen_interval_width_ratio_to_global']:.3f}",
                f"- UQ support: `{row['uq_support_gate']}`",
                f"- Risk–error Spearman: "
                f"{row['risk_error_spearman']:.3f}",
                f"- Selective support: `{row['selective_support_gate']}`",
                "",
            ]
        report += [
            "## أهم الملفات",
            "- `tables/STAGE3B_EXTERNAL_RESULTS.xlsx`",
            "- `tables/EXTERNAL_POINT_METRICS.csv`",
            "- `tables/EXTERNAL_INTERVAL_METRICS.csv`",
            "- `tables/WILCOXON_HOLM_COMPARISONS.csv`",
            "- `tables/EXTERNAL_SUPPORT_GATES.csv`",
            "- `data/EXTERNAL_FROZEN_PREDICTIONS.csv`",
            "",
            "## حد الادعاء",
            "هذه النتيجة لا تغلق حاجة البحث إلى same-regime clear-wood external lockbox.",
        ]
        (OUT/"EXTERNAL_TRANSPORT_REPORT_AR.md").write_text(
            "\n".join(report)+"\n", encoding="utf-8"
        )

        write_json(OUT/"OUTPUT_VERIFICATION.json", {
            "verification_passed": True,
            "stage0f_verified": True,
            "stage3a_fix2_verified": True,
            "lockbox_hash_verified": True,
            "frozen_protocol_verified": True,
            "external_labels_used_for_training": False,
            "external_labels_used_for_calibration": False,
            "method_parameters_changed": False,
            "wilcoxon_holm_completed": True,
            "bootstrap_iterations": BOOT,
            "same_regime_claim_prevented": True
        })
        write_json(OUT/"CHECKPOINT.json", {
            "completed_steps": [
                "input_and_hash_verification",
                "registered_feature_mapping",
                "three_source_final_model_fit",
                "development_only_interval_calibration",
                "external_point_evaluation",
                "external_interval_evaluation",
                "wilcoxon_holm_comparison",
                "bootstrap_confidence_intervals",
                "selective_risk_and_spacing_audit"
            ],
            "timestamp": now()
        })
        write_json(OUT/"STATUS.json", {
            "stage": "STAGE3B", "version": "1.0.0",
            "status": "completed", "decision": decision,
            "completed_at": now()
        })
        write_json(OUT/"SUCCESS_MARKER.json", {
            "success": True, "stage": "STAGE3B", "version": "1.0.0",
            "decision": decision, "timestamp": now()
        })
        write_json(OUT/"ENVIRONMENT_SNAPSHOT.json", {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__
        })
        (OUT/"ERROR_LOG.txt").write_text("", encoding="utf-8")
        (OUT/"RUN_LOG.txt").write_text(
            "\n".join(LOG_LINES)+"\n", encoding="utf-8"
        )

        required = [
            "EXTERNAL_TRANSPORT_REPORT_AR.md",
            "SCIENTIFIC_STAGE3B_DECISION.json",
            "tables/STAGE3B_EXTERNAL_RESULTS.xlsx",
            "tables/EXTERNAL_POINT_METRICS.csv",
            "tables/EXTERNAL_INTERVAL_METRICS.csv",
            "tables/WILCOXON_HOLM_COMPARISONS.csv",
            "tables/EXTERNAL_SUPPORT_GATES.csv",
            "data/EXTERNAL_FROZEN_PREDICTIONS.csv",
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
                "FILE_MANIFEST.csv", "HASHES_SHA256.csv",
                "UPLOAD_THIS_STAGE3B_RESULT.zip"
            }:
                continue
            manifest_rows.append({
                "relative_path": rel,
                "size_bytes": path.stat().st_size
            })
            hash_rows.append({
                "relative_path": rel,
                "sha256": sha256(path)
            })
        pd.DataFrame(manifest_rows).to_csv(
            OUT/"FILE_MANIFEST.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(hash_rows).to_csv(
            OUT/"HASHES_SHA256.csv", index=False, encoding="utf-8-sig"
        )

        upload = OUT/"UPLOAD_THIS_STAGE3B_RESULT.zip"
        with zipfile.ZipFile(upload, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(OUT.rglob("*")):
                if path.is_file() and path != upload and "_work" not in path.parts:
                    z.write(path, path.relative_to(OUT))

        log(f"Completed Stage3B. Decision: {decision}")
        (OUT/"RUN_LOG.txt").write_text(
            "\n".join(LOG_LINES)+"\n", encoding="utf-8"
        )
        return 0

    except Exception as exc:
        ERROR_LINES.append(traceback.format_exc())
        write_json(OUT/"STATUS.json", {
            "stage": "STAGE3B", "status": "failed",
            "error": str(exc), "completed_at": now()
        })
        write_json(OUT/"SUCCESS_MARKER.json", {
            "success": False, "stage": "STAGE3B",
            "error": str(exc), "timestamp": now()
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
