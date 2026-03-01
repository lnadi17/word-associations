"""FA-style cleaning pipeline for free association data.

Ported from LWOW reproducibility scripts (FA_Functions, FA_data_Cleaning)
to produce processed datasets in cue,R1,R2,R3 format aligned with LWOW.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
except ImportError:
    wn = None  # type: ignore
    WordNetLemmatizer = None  # type: ignore

_COLS = ["cue", "R1", "R2", "R3"]
_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
_MISSING_DICT_CACHE = _CACHE_DIR / "fa_missing_dict.json"
_WN_WORD_SET_CACHE = _CACHE_DIR / "fa_wn_word_set.json"


def _get_lemmatizer() -> Any:
    if WordNetLemmatizer is None:
        raise ImportError("nltk required for fa_cleaning. Install with: pip install nltk")
    return WordNetLemmatizer()


def load_spelling_dict(path: str | Path) -> Dict[str, str]:
    """Load tab-separated wrong->correct spelling mappings. Returns {wrong.lower(): correct.lower()}."""
    result: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" in line:
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    wrong, correct = parts[0].strip(), parts[1].strip()
                    if wrong and correct:
                        result[wrong.lower()] = correct.lower()
    return result


def _build_wn_data() -> tuple[Dict[str, str], set]:
    """Single WordNet pass: build missing_dict and wn_word_set. Used by both caching functions."""
    if wn is None:
        raise ImportError("nltk.corpus.wordnet required")
    wn_lower: List[str] = []
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            wn_lower.append(str(lemma.name()).lower())
    wn_lower_set = set(wn_lower)  # O(1) membership
    no_spaces = {x.replace("_", ""): x.replace("_", " ") for x in wn_lower}
    no_hyphens = {x.replace("-", ""): x for x in wn_lower}
    only_no_spaces = {k: v for k, v in no_spaces.items() if k not in wn_lower_set}
    only_no_hyphens = {k: v for k, v in no_hyphens.items() if k not in wn_lower_set}
    missing_dict = dict(only_no_spaces)
    missing_dict.update(only_no_hyphens)
    return missing_dict, wn_lower_set


def build_missing_dict(use_cache: bool = True) -> Dict[str, str]:
    """Build WordNet-based mapping for words missing spaces or hyphens. Cached to speed up repeated runs."""
    if use_cache and _MISSING_DICT_CACHE.exists():
        try:
            with open(_MISSING_DICT_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    missing_dict, wn_set = _build_wn_data()
    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_MISSING_DICT_CACHE, "w", encoding="utf-8") as f:
            json.dump(missing_dict, f)
        with open(_WN_WORD_SET_CACHE, "w", encoding="utf-8") as f:
            json.dump(sorted(wn_set), f)
    return missing_dict


def get_wn_word_set(use_cache: bool = True) -> set:
    """Set of valid WordNet words (lowercase, underscores). For fast wn_filter lookups. Cached."""
    if use_cache and _WN_WORD_SET_CACHE.exists():
        try:
            with open(_WN_WORD_SET_CACHE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    _, wn_set = _build_wn_data()
    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_WN_WORD_SET_CACHE, "w", encoding="utf-8") as f:
            json.dump(sorted(wn_set), f)
    return wn_set


def _lemmatize_cell(x: str, lemmatizer: Any) -> str:
    if not x or not isinstance(x, str):
        return ""
    out = lemmatizer.lemmatize(x)
    if out == "men":
        return "man"
    out = out.replace("hands", "hand")
    return out


def derive_unq_cues(
    df: pd.DataFrame,
    spelling_dict: Dict[str, str],
    missing_dict: Dict[str, str],
    lemmatizer: Any,
) -> List[str]:
    """Derive unique cues from dataframe with same normalization as FA pipeline."""
    cues = df["cue"].dropna().astype(str).str.lower().unique()
    cues = [c.replace("_", " ") for c in cues]
    cues = [spelling_dict.get(c, c) for c in cues]
    cues = [missing_dict.get(c, c) for c in cues]
    cues = [_lemmatize_cell(c, lemmatizer) for c in cues]
    return list(set(c for c in cues if c))


def na2blank(df: pd.DataFrame) -> pd.DataFrame:
    """Convert non-strings to blanks."""
    out = df.copy()
    for col in _COLS:
        out[col] = [x if isinstance(x, str) else "" for x in out[col]]
    return out


def lowercase(df: pd.DataFrame) -> pd.DataFrame:
    """Make all cue and response columns lowercase."""
    out = df.copy()
    for col in _COLS:
        out[col] = [x.lower() for x in out[col]]
    return out


def remove_underscore(df: pd.DataFrame) -> pd.DataFrame:
    """Replace underscores with spaces."""
    out = df.copy()
    for col in _COLS:
        out[col] = [x.replace("_", " ") for x in out[col]]
    return out


def remove_resp_articles(df: pd.DataFrame, unq_cues: List[str]) -> pd.DataFrame:
    """Remove leading 'a ', 'an ', 'the ', 'to ' from responses unless the response is a cue."""
    out = df.copy()
    cue_set = set(unq_cues)
    for col in ["R1", "R2", "R3"]:
        for prefix in ["a ", "an ", "the ", "to "]:
            mask = (
                out[col].astype(str).str.startswith(prefix)
                & (~out[col].isin(cue_set))
            )
            out.loc[mask, col] = out.loc[mask, col].astype(str).str[len(prefix) :]
    return out


def add_space_or_hyphen(df: pd.DataFrame, missing_dict: Dict[str, str]) -> pd.DataFrame:
    """Map words missing spaces/hyphens to WordNet forms."""
    out = df.copy()
    for col in _COLS:
        out[col] = out[col].map(lambda x: missing_dict.get(x, x) if isinstance(x, str) else "")
    return out


def spelling(df: pd.DataFrame, spelling_dict: Dict[str, str]) -> pd.DataFrame:
    """Correct spelling using mapping dictionary."""
    out = df.copy()
    for col in _COLS:
        out[col] = out[col].map(lambda x: spelling_dict.get(x, x) if isinstance(x, str) else "")
    return out


def lemmatization(df: pd.DataFrame, lemmatizer: Any) -> pd.DataFrame:
    """Lemmatize and apply manual corrections (men->man, hands->hand)."""
    out = df.copy()
    for col in _COLS:
        out[col] = [_lemmatize_cell(x, lemmatizer) for x in out[col]]
    return out


def remove_cue_resp(df: pd.DataFrame) -> pd.DataFrame:
    """Remove responses equal to the cue."""
    out = df.copy()
    for col in ["R1", "R2", "R3"]:
        out[col] = np.where(out[col] == out["cue"], "", out[col])
    return out


def remove_dupe_resp(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate responses within a row (R3=R1/R2, R2=R1)."""
    out = df.copy()
    out["R3"] = np.where(
        (out["R3"] == out["R1"]) | (out["R3"] == out["R2"]), "", out["R3"]
    )
    out["R2"] = np.where(out["R2"] == out["R1"], "", out["R2"])
    return out


def shift_resp(df: pd.DataFrame) -> pd.DataFrame:
    """Shift responses left so blanks are on the right."""
    out = df.copy()
    # _ _ X -> X _ _
    out["R1"] = np.where(
        (out["R1"] == "") & (out["R2"] == "") & (out["R3"] != ""),
        out["R3"],
        out["R1"],
    )
    out["R3"] = np.where(out["R1"] == out["R3"], "", out["R3"])
    # _ X _ -> X _ _
    out["R1"] = np.where(
        (out["R1"] == "") & (out["R2"] != "") & (out["R3"] == ""),
        out["R2"],
        out["R1"],
    )
    out["R2"] = np.where(out["R1"] == out["R2"], "", out["R2"])
    # _ X X -> X _ X
    out["R1"] = np.where(
        (out["R1"] == "") & (out["R2"] != "") & (out["R3"] != ""),
        out["R2"],
        out["R1"],
    )
    out["R2"] = np.where(out["R1"] == out["R2"], "", out["R2"])
    # X _ X -> X X _
    out["R2"] = np.where(
        (out["R1"] != "") & (out["R2"] == "") & (out["R3"] != ""),
        out["R3"],
        out["R2"],
    )
    out["R3"] = np.where(out["R2"] == out["R3"], "", out["R3"])
    return out


def sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by cue, R1, R2, R3."""
    out = df[_COLS].copy()
    return out.sort_values(by=_COLS)


def cue100(
    df: pd.DataFrame, unq_cues: List[str], seed: int = 30, repetitions: int = 100
) -> pd.DataFrame:
    """Align to exactly `repetitions` rows per cue. Sample down or pad with blanks."""
    rng = random.Random(seed)
    unq_set = set(unq_cues)
    df = df[df["cue"].isin(unq_set)].copy()
    cue_counts = df["cue"].value_counts()
    over = {c: n for c, n in cue_counts.items() if n > repetitions}
    under = {c: n for c, n in cue_counts.items() if n < repetitions}
    missing = unq_set - set(df["cue"])

    if over:
        rows_to_drop: List[int] = []
        for c, count in over.items():
            idx = df[df["cue"] == c].index.tolist()
            surplus = count - repetitions
            rows_to_drop.extend(rng.sample(idx, surplus))
        df = df.drop(rows_to_drop)

    extra_rows: List[Dict[str, str]] = []
    if under:
        for c, count in under.items():
            deficit = repetitions - count
            extra_rows.extend([{"cue": c, "R1": "", "R2": "", "R3": ""}] * deficit)
    if missing:
        for c in missing:
            extra_rows.extend([{"cue": c, "R1": "", "R2": "", "R3": ""}] * repetitions)
    if extra_rows:
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    return df


def raw_rows_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert raw API rows (cue,trial,parsed) to DataFrame with cue,R1,R2,R3."""
    records: List[Dict[str, str]] = []
    for row in rows:
        cue = str(row.get("cue", "")).strip()
        parsed = row.get("parsed", "")
        tokens = [t.strip() for t in str(parsed).split("|") if t.strip()]
        r1 = tokens[0] if len(tokens) > 0 else ""
        r2 = tokens[1] if len(tokens) > 1 else ""
        r3 = tokens[2] if len(tokens) > 2 else ""
        records.append({"cue": cue, "R1": r1, "R2": r2, "R3": r3})
    return pd.DataFrame(records)


def cleaning_pipeline(
    rows: List[Dict[str, Any]],
    spelling_dict: Optional[Dict[str, str]] = None,
    spelling_dict_path: Optional[str | Path] = None,
    seed: int = 30,
    repetitions_per_cue: int = 100,
) -> pd.DataFrame:
    """Run full FA cleaning pipeline. Returns DataFrame with cue,R1,R2,R3."""
    df = raw_rows_to_dataframe(rows)
    if df.empty:
        return pd.DataFrame(columns=_COLS)

    spelling_dict = spelling_dict or {}
    if spelling_dict_path and Path(spelling_dict_path).exists():
        spelling_dict = load_spelling_dict(spelling_dict_path)

    missing_dict = build_missing_dict()
    lemmatizer = _get_lemmatizer()

    df = na2blank(df)
    df = lowercase(df)
    df = remove_underscore(df)
    unq_cues = derive_unq_cues(df, spelling_dict, missing_dict, lemmatizer)
    df = remove_resp_articles(df, unq_cues)
    df = add_space_or_hyphen(df, missing_dict)
    df = spelling(df, spelling_dict)
    df = lemmatization(df, lemmatizer)
    df = cue100(df, unq_cues, seed=seed, repetitions=repetitions_per_cue)
    df = remove_cue_resp(df)
    df = remove_dupe_resp(df)
    df = shift_resp(df)
    df = sort_columns(df)
    return df
