#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr


Triplet = Tuple[str, str, str]


def normalize_df(df: pd.DataFrame, normalize_rows: bool = False) -> pd.DataFrame:
    cols = list(df.columns)[1:]
    mat = df.copy().drop(columns=["node"]).to_numpy(dtype=float)
    # Normalize columns (L2), matching sklearn.preprocessing.normalize(..., axis=0)
    col_norms = np.linalg.norm(mat, axis=0, keepdims=True)
    col_norms[col_norms == 0.0] = 1.0
    mat_norm = mat / col_norms
    if normalize_rows:
        # Normalize rows (L2), matching sklearn.preprocessing.normalize(..., axis=1)
        row_norms = np.linalg.norm(mat_norm, axis=1, keepdims=True)
        row_norms[row_norms == 0.0] = 1.0
        mat_norm = mat_norm / row_norms
    out = pd.DataFrame(mat_norm, columns=cols, index=list(df["node"]))
    return out


def activation_dict(df_norm: pd.DataFrame, triplets: Iterable[Triplet]) -> Dict[Triplet, Tuple[float, float, float]]:
    out: Dict[Triplet, Tuple[float, float, float]] = {}
    for tgt, rel, un in triplets:
        rel_act = float(df_norm[rel].loc[tgt])
        un_act = float(df_norm[un].loc[tgt])
        out[(tgt, rel, un)] = (rel_act, un_act, rel_act - un_act)
    return out


def matrices_gender(df_norm: pd.DataFrame, rows: List[str], primes_f: List[str], primes_m: List[str]) -> pd.DataFrame:
    return df_norm[primes_f + primes_m].loc[rows]


def _steiger_z(r_12: float, r_13: float, r_23: float, n: int) -> Tuple[float, float]:
    """Steiger's Z test for comparing two dependent correlations (r_12 vs r_13, variable 1 in common).
    Returns (z_stat, p_value). p is one-tailed: P(Z > z) when r_12 > r_13."""
    if n < 4:
        return float("nan"), float("nan")
    z12 = math.atanh(r_12)
    z13 = math.atanh(r_13)
    d = 1 - r_12**2 - r_13**2 - r_23**2 + 2 * r_12 * r_13 * r_23
    denom = (1 - r_23**2)
    if denom <= 0 or d <= 0:
        return float("nan"), float("nan")
    var_diff = 2 * d / ((n - 3) * denom)
    if var_diff <= 0:
        return float("nan"), float("nan")
    z = (z12 - z13) / math.sqrt(var_diff)
    p = 1 - float(stats.norm.cdf(z))
    return z, p


def _wilcoxon_effect(data: List[float], alternative: str) -> Tuple[float, float]:
    if len(data) == 0:
        return float("nan"), float("nan")
    res = stats.wilcoxon(data, alternative=alternative, method="approx")
    p = float(res.pvalue)
    z = float(res.zstatistic) if hasattr(res, "zstatistic") else float("nan")
    effect = z / math.sqrt(len(data)) if not np.isnan(z) else float("nan")
    return p, effect


def _load_triplets(path: Path) -> Tuple[List[Triplet], Dict[Triplet, Tuple[float, float, float]]]:
    df = pd.read_csv(path)
    required = {"Target", "Related Prime", "Unrelated Prime", "Target-Related RT", "Target-Unrelated RT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    triplets: List[Triplet] = []
    rt_dict: Dict[Triplet, Tuple[float, float, float]] = {}
    for row in df.itertuples(index=False):
        tgt = str(getattr(row, "Target")).strip().lower()
        rel = str(getattr(row, "_1")).strip().lower() if hasattr(row, "_1") else str(getattr(row, "Related_Prime")).strip().lower()
        un = str(getattr(row, "_2")).strip().lower() if hasattr(row, "_2") else str(getattr(row, "Unrelated_Prime")).strip().lower()
        # Access by column names to avoid tuple field name quirks with spaces.
        rel_rt = float(getattr(row, "_3")) if hasattr(row, "_3") else float(getattr(row, "Target_Related_RT"))
        un_rt = float(getattr(row, "_4")) if hasattr(row, "_4") else float(getattr(row, "Target_Unrelated_RT"))
        trip = (tgt, rel, un)
        triplets.append(trip)
        rt_dict[trip] = (rel_rt, un_rt, un_rt - rel_rt)
    return triplets, rt_dict


def _load_triplets_robust(path: Path) -> Tuple[List[Triplet], Dict[Triplet, Tuple[float, float, float]]]:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    target_col = cols.get("target")
    rel_col = cols.get("related prime")
    un_col = cols.get("unrelated prime")
    tr_col = cols.get("target-related rt")
    tu_col = cols.get("target-unrelated rt")
    needed = [target_col, rel_col, un_col, tr_col, tu_col]
    if any(c is None for c in needed):
        raise ValueError(
            f"{path} must include columns: Target, Related Prime, Unrelated Prime, Target-Related RT, Target-Unrelated RT"
        )
    triplets: List[Triplet] = []
    rt_dict: Dict[Triplet, Tuple[float, float, float]] = {}
    for _, r in df.iterrows():
        tgt = str(r[target_col]).strip().lower()
        rel = str(r[rel_col]).strip().lower()
        un = str(r[un_col]).strip().lower()
        rel_rt = float(r[tr_col])
        un_rt = float(r[tu_col])
        trip = (tgt, rel, un)
        triplets.append(trip)
        rt_dict[trip] = (rel_rt, un_rt, un_rt - rel_rt)
    return triplets, rt_dict


def _get_models(matrices_dir: Path) -> List[str]:
    ldt = {p.stem[: -len("_LDT")] for p in matrices_dir.glob("*_LDT.csv")}
    gender = {p.stem[: -len("_gender")] for p in matrices_dir.glob("*_gender.csv")}
    models = sorted(ldt.intersection(gender))
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FA LDT/gender statistical tests.")
    parser.add_argument("--analyses-dir", default="data/LDT_analyses")
    parser.add_argument("--human-model", default="Humans")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    analyses_dir = Path(args.analyses_dir)
    matrices_dir = analyses_dir / "FA_matrices"
    stats_dir = analyses_dir / "Statistical_tests"
    stats_dir.mkdir(parents=True, exist_ok=True)

    triplets_path = analyses_dir / "LDT_50_triplets.csv"
    gender_targets_path = analyses_dir / "gender_targets.csv"
    gender_primes_path = analyses_dir / "gender_primes.csv"

    if not matrices_dir.exists():
        raise SystemExit(f"Missing FA matrices directory: {matrices_dir}")
    if not triplets_path.exists():
        raise SystemExit(f"Missing LDT triplets file: {triplets_path}")
    if not gender_targets_path.exists():
        raise SystemExit(f"Missing gender targets file: {gender_targets_path}")
    if not gender_primes_path.exists():
        raise SystemExit(f"Missing gender primes file: {gender_primes_path}")

    models = _get_models(matrices_dir)
    if not models:
        raise SystemExit(f"No matching *_LDT.csv and *_gender.csv files found in {matrices_dir}")
    if args.human_model not in models:
        raise SystemExit(f"Human model '{args.human_model}' not found. Available: {models}")

    if args.verbose:
        print("Models:", ", ".join(models))

    triplets, ldt_rt_dict = _load_triplets_robust(triplets_path)

    ldt_dfs = {m: pd.read_csv(matrices_dir / f"{m}_LDT.csv") for m in models}
    gender_dfs = {m: pd.read_csv(matrices_dir / f"{m}_gender.csv") for m in models}

    # Node intersection across all models from gender matrices.
    node_sets = [set(df["node"].astype(str).str.lower()) for df in gender_dfs.values()]
    node_int_full = set.intersection(*node_sets) if node_sets else set()

    gender_prime_pairs = [
        ("woman", "man"),
        ("female", "male"),
        ("mother", "father"),
        ("girl", "boy"),
        ("feminine", "masculine"),
    ]
    primes_f = [a for a, _ in gender_prime_pairs]
    primes_m = [b for _, b in gender_prime_pairs]

    targets = pd.read_csv(gender_targets_path)
    tgts_f = [str(x).strip().lower() for x in targets["Female"] if str(x).strip().lower() in node_int_full]
    tgts_m = [str(x).strip().lower() for x in targets["Male"] if str(x).strip().lower() in node_int_full]

    # LDT activation stats by model.
    ldt_al_dicts = {}
    for model in models:
        norm = normalize_df(ldt_dfs[model], normalize_rows=True)
        ldt_al_dicts[model] = activation_dict(norm, triplets)

    wilcoxon_ldt = {}
    for model, d in ldt_al_dicts.items():
        rel_vals = [x[0] for x in d.values()]
        un_vals = [x[1] for x in d.values()]
        diff = [a - b for a, b in zip(rel_vals, un_vals)]
        p, effect = _wilcoxon_effect(diff, alternative="greater")
        wilcoxon_ldt[model] = {"p": p, "effect": effect}
    pd.DataFrame(wilcoxon_ldt).to_csv(stats_dir / "wilcoxonLDT.csv", index=True)

    corrs_ldt = {}
    rt_rel = [x[0] for x in ldt_rt_dict.values()]
    rt_un = [x[1] for x in ldt_rt_dict.values()]
    for model, d in ldt_al_dicts.items():
        al_rel = [x[0] for x in d.values()]
        al_un = [x[1] for x in d.values()]
        r, p = spearmanr(np.concatenate([al_rel, al_un]), np.concatenate([rt_rel, rt_un]))
        corrs_ldt[model] = {"r": r, "p": p}
    pd.DataFrame(corrs_ldt).to_csv(stats_dir / "corrLDT.csv", index=True)

    # Gender activation matrices by model.
    mat_gender = {}
    for model in models:
        norm = normalize_df(gender_dfs[model], normalize_rows=True)
        mat_f = matrices_gender(norm, tgts_f, primes_f, primes_m)
        mat_m = matrices_gender(norm, tgts_m, primes_f, primes_m)
        mat_gender[model] = {
            "Female-related targets": mat_f,
            "Male-related targets": mat_m,
        }

    wilcoxon_f = {}
    wilcoxon_m = {}
    for model in models:
        test_res = {}
        for target_group, alt in [("Female-related targets", "greater"), ("Male-related targets", "less")]:
            data1 = mat_gender[model][target_group][primes_f].values.flatten()
            data2 = mat_gender[model][target_group][primes_m].values.flatten()
            diff = [a - b for a, b in zip(data1, data2)]
            p, effect = _wilcoxon_effect(diff, alternative=alt)
            test_res[target_group] = {"p": p, "effect": effect}
        wilcoxon_f[model] = test_res["Female-related targets"]
        wilcoxon_m[model] = test_res["Male-related targets"]
    # Keep historical filenames used in current repo.
    pd.DataFrame(wilcoxon_f).to_csv(stats_dir / "wilxoconGenderF.csv", index=True)
    pd.DataFrame(wilcoxon_m).to_csv(stats_dir / "wilxoconGenderM.csv", index=True)

    # Human-vs-model correlations for gender activation.
    models_corr_f = {}
    models_corr_m = {}
    for model in models:
        if model == args.human_model:
            continue
        mat1f = np.concatenate(mat_gender[args.human_model]["Female-related targets"].values)
        mat2f = np.concatenate(mat_gender[model]["Female-related targets"].values)
        rf, pf = spearmanr(mat1f, mat2f)
        models_corr_f[model] = {"r": rf, "p": pf}

        mat1m = np.concatenate(mat_gender[args.human_model]["Male-related targets"].values)
        mat2m = np.concatenate(mat_gender[model]["Male-related targets"].values)
        rm, pm = spearmanr(mat1m, mat2m)
        models_corr_m[model] = {"r": rm, "p": pm}
    pd.DataFrame(models_corr_f).to_csv(stats_dir / "corrModelsF.csv", index=True)
    pd.DataFrame(models_corr_m).to_csv(stats_dir / "corrModelsM.csv", index=True)

    # Pairwise correlation matrices (all models including Humans).
    llms = [m for m in models if m != args.human_model]
    for target_type in ["Female-related targets", "Male-related targets"]:
        flat = {m: np.concatenate(mat_gender[m][target_type].values) for m in models}
        n = len(flat[models[0]])
        r_mat = pd.DataFrame(index=models, columns=models, dtype=float)
        for a in models:
            for b in models:
                if a == b:
                    r_mat.loc[a, b] = 1.0
                elif a < b:
                    r, _ = spearmanr(flat[a], flat[b])
                    r_mat.loc[a, b] = r
                    r_mat.loc[b, a] = r
                # else filled by symmetry
        suffix = "F" if "Female" in target_type else "M"
        r_mat.to_csv(stats_dir / f"corr_pairwise_{suffix}.csv")

        # Steiger Z: for each LLM pair, test if one is more aligned with Humans.
        steiger_rows = []
        for i, a in enumerate(llms):
            for b in llms:
                if a >= b:
                    continue
                r_ha = float(models_corr_f[a]["r"] if "Female" in target_type else models_corr_m[a]["r"])
                r_hb = float(models_corr_f[b]["r"] if "Female" in target_type else models_corr_m[b]["r"])
                r_ab, _ = spearmanr(flat[a], flat[b])
                z, p = _steiger_z(r_ha, r_hb, r_ab, n)
                steiger_rows.append({
                    "model_a": a, "model_b": b,
                    "r_Human_a": r_ha, "r_Human_b": r_hb, "r_a_b": r_ab,
                    "z": z, "p_a_more_aligned": p,
                })
        steiger_df = pd.DataFrame(steiger_rows)
        steiger_df.to_csv(stats_dir / f"steiger_{suffix}.csv", index=False)

    if args.verbose:
        print(f"Wrote: {stats_dir / 'wilcoxonLDT.csv'}")
        print(f"Wrote: {stats_dir / 'corrLDT.csv'}")
        print(f"Wrote: {stats_dir / 'wilxoconGenderF.csv'}")
        print(f"Wrote: {stats_dir / 'wilxoconGenderM.csv'}")
        print(f"Wrote: {stats_dir / 'corrModelsF.csv'}")
        print(f"Wrote: {stats_dir / 'corrModelsM.csv'}")
        print(f"Wrote: {stats_dir / 'corr_pairwise_F.csv'}, {stats_dir / 'corr_pairwise_M.csv'}")
        print(f"Wrote: {stats_dir / 'steiger_F.csv'}, {stats_dir / 'steiger_M.csv'}")


if __name__ == "__main__":
    main()
