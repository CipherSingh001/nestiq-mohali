# =============================================================
# PHASE 5 — recommender.py
# Smart Real Estate Advisor — Mohali
# =============================================================
# SETUP:
#   pip install pandas numpy scikit-learn xgboost joblib groq
#
# RUN (standalone test):
#   python recommender.py
#
# USED BY:
#   app.py → results_df, filters, error = recommend_properties(query)
# =============================================================

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

# ── Paths (relative to this file — works from any cwd) ───────
BASE_DIR    = Path(__file__).parent
DATA_PATH   = BASE_DIR / "data"   / "mohali_properties_final.csv"
MODEL_PATH  = BASE_DIR / "models" / "model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "models" / "features.pkl"

# ── Furnishing encoding shared across Phases 3 and 5 ─────────
FURNISHING_MAP = {
    "Unfurnished":     0,
    "Semi":            1,
    "Semi-Furnished":  1,
    "semi-furnished":  1,
    "Furnished":       2,
}

# ─────────────────────────────────────────────────────────────
# LOAD RESOURCES (cached so they load once per process)
# ─────────────────────────────────────────────────────────────
_cache = {}

def _load_resources():
    """Load CSV, model, scaler, and features once and cache them."""
    if _cache:
        return _cache

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")

    _cache["df"]       = pd.read_csv(DATA_PATH)
    _cache["model"]    = joblib.load(MODEL_PATH)
    _cache["scaler"]   = joblib.load(SCALER_PATH)
    _cache["features"] = joblib.load(FEATURES_PATH)   # list of 29 feature names, in order

    print(f"  ✅ Loaded {len(_cache['df'])} properties, {len(_cache['features'])} model features")
    return _cache


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# Must reproduce the exact same 29 features train_model.py used
# ─────────────────────────────────────────────────────────────
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate all 29 features required by the XGBoost model.
    Input df must have the raw columns from the CSV.
    Returns a copy with all engineered columns added.
    """
    df = df.copy()

    # ── One-hot encode Property_Type ─────────────────────────
    df["Property_Type_House"] = (df["Property_Type"] == "House").astype(int)
    df["Property_Type_Villa"] = (df["Property_Type"] == "Villa").astype(int)

    # ── Ordinal encode Furnishing (handles object, string[pyarrow], and already-int) ──
    if not pd.api.types.is_numeric_dtype(df["Furnishing"]):
        df["Furnishing"] = df["Furnishing"].astype(str).map(FURNISHING_MAP).fillna(0).astype(int)

    # ── 11 engineered interaction features ───────────────────
    df["Area_sqft_sq"]        = df["Area_sqft"] ** 2
    df["log_Area"]            = np.log1p(df["Area_sqft"])      # log1p avoids log(0)
    df["Area_x_BHK"]          = df["Area_sqft"] * df["BHK"]
    df["Area_x_Villa"]        = df["Area_sqft"] * df["Property_Type_Villa"]
    df["loc_x_area"]          = df["Location_Score"] * df["Area_sqft"]
    df["amenity_x_area"]      = df["Amenities_Score"] * df["Area_sqft"]
    df["BHK_x_Floor"]         = df["BHK"] * df["Floor"]
    df["loc_x_amenity"]       = df["Location_Score"] * df["Amenities_Score"]
    df["dist_airport_x_area"] = df["Distance_Airport_km"] * df["Area_sqft"]
    df["age_x_area"]          = df["Age"] * df["Area_sqft"]

    return df


def _build_feature_matrix(df: pd.DataFrame, features: list) -> np.ndarray:
    """Select and order columns to match the training feature list."""
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features after engineering: {missing}")
    return df[features].values


# ─────────────────────────────────────────────────────────────
# FILTER ENGINE
# ─────────────────────────────────────────────────────────────
def filter_properties(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply structured filters (from query_processor.parse_query) to the dataset.
    Returns a filtered DataFrame (may be empty if no matches).
    """
    filtered = df.copy()
    n_start = len(filtered)

    # BHK
    if filters.get("min_bhk") is not None:
        filtered = filtered[filtered["BHK"] >= filters["min_bhk"]]
    if filters.get("max_bhk") is not None:
        filtered = filtered[filtered["BHK"] <= filters["max_bhk"]]

    # Price
    if filters.get("min_price") is not None:
        filtered = filtered[filtered["Price_Lakh"] >= filters["min_price"]]
    if filters.get("max_price") is not None:
        filtered = filtered[filtered["Price_Lakh"] <= filters["max_price"]]

    # Area
    if filters.get("min_area") is not None:
        filtered = filtered[filtered["Area_sqft"] >= filters["min_area"]]
    if filters.get("max_area") is not None:
        filtered = filtered[filtered["Area_sqft"] <= filters["max_area"]]

    # Sector
    if filters.get("sector") is not None:
        filtered = filtered[filtered["Sector"] == filters["sector"]]

    # Property type
    if filters.get("property_type") is not None:
        filtered = filtered[filtered["Property_Type"].str.lower() == filters["property_type"].lower()]

    # Furnishing — query_processor returns string label; compare against CSV string or encoded int
    if filters.get("furnishing") is not None:
        furn_val = filters["furnishing"]
        if not pd.api.types.is_numeric_dtype(filtered["Furnishing"]):
            # CSV stores strings: "Furnished", "Semi-Furnished", "Unfurnished"
            # Normalise "Semi" → "Semi-Furnished" for matching
            if furn_val == "Semi":
                furn_val = "Semi-Furnished"
            filtered = filtered[filtered["Furnishing"].astype(str) == furn_val]
        else:
            # Already encoded as int 0/1/2
            target_int = FURNISHING_MAP.get(furn_val, -1)
            if target_int >= 0:
                filtered = filtered[filtered["Furnishing"] == target_int]

    # Nearby amenities
    if filters.get("near_hospital") is True:
        filtered = filtered[filtered["Near_Hospital"] == 1]
    if filters.get("near_park") is True:
        filtered = filtered[filtered["Near_Park"] == 1]
    if filters.get("near_mall") is True:
        filtered = filtered[filtered["Near_Mall"] == 1]

    # Property age
    if filters.get("max_age") is not None:
        filtered = filtered[filtered["Age"] <= filters["max_age"]]

    n_end = len(filtered)
    print(f"  🔍 Filtered {n_start} → {n_end} properties ({n_start - n_end} removed)")
    return filtered.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────
def score_properties(df: pd.DataFrame, model, scaler, features: list) -> pd.DataFrame:
    """
    Score each property using:
      - ML predicted price → value_score  (underpriced = higher score)
      - Location_Score    → location component
      - Amenities_Score   → amenity component
      - Investment_Score  → if column exists in original data

    final_score = 0.40 × value_score_norm
                + 0.30 × location_norm
                + 0.20 × amenity_norm
                + 0.10 × investment_norm
    """
    df = df.copy()

    # ── Step 1: build feature matrix with engineered features ─
    df_eng = add_engineered_features(df)
    X = _build_feature_matrix(df_eng, features)
    X_scaled = scaler.transform(X)

    # ── Step 2: ML price prediction ───────────────────────────
    df["predicted_price"] = model.predict(X_scaled)

    # ── Step 3: value score — how underpriced is this? ────────
    # Positive = model thinks it should cost MORE than listed (good deal)
    # Clipped to [-1, 1] to avoid extreme outliers dominating
    df["value_score"] = (df["predicted_price"] - df["Price_Lakh"]) / df["predicted_price"].clip(lower=1)
    df["value_score"] = df["value_score"].clip(-1, 1)

    # ── Step 4: normalise each component to [0, 1] ───────────
    def norm(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    value_norm     = norm(df["value_score"])
    location_norm  = norm(df["Location_Score"])
    amenity_norm   = norm(df["Amenities_Score"])

    # Investment_Score may have been dropped during training to prevent leakage,
    # but it's fine to use as a ranking signal here (we're not training)
    if "Investment_Score" in df.columns:
        invest_norm = norm(df["Investment_Score"])
    else:
        invest_norm = value_norm   # fallback

    # ── Step 5: weighted final score ─────────────────────────
    df["final_score"] = (
        0.40 * value_norm
      + 0.30 * location_norm
      + 0.20 * amenity_norm
      + 0.10 * invest_norm
    )

    return df


# ─────────────────────────────────────────────────────────────
# FALLBACK: relax filters one by one when nothing matches
# ─────────────────────────────────────────────────────────────
_RELAXATION_ORDER = [
    "sector", "property_type", "furnishing",
    "near_hospital", "near_park", "near_mall",
    "max_age", "min_area", "max_area",
    "min_price", "max_price",
    "min_bhk", "max_bhk",
]

def _relax_filters(filters: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, dict, str]:
    """
    Progressively remove filters until at least 5 results are found.
    Returns (filtered_df, relaxed_filters, message).
    """
    relaxed = dict(filters)
    removed = []

    for key in _RELAXATION_ORDER:
        if len(df) >= 5:
            break
        if relaxed.get(key) is not None and relaxed.get(key) is not True and relaxed.get(key) is not False:
            relaxed.pop(key, None)
            removed.append(key)
            df = filter_properties(
                pd.read_csv(DATA_PATH), relaxed
            )
        elif relaxed.get(key) is True:
            relaxed[key] = None
            removed.append(key)
            df = filter_properties(
                pd.read_csv(DATA_PATH), relaxed
            )

    msg = ""
    if removed:
        msg = f"⚠️ No exact matches. Relaxed: {', '.join(removed)} to show nearby results."

    return df, relaxed, msg


# ─────────────────────────────────────────────────────────────
# PUBLIC API — main entry point
# ─────────────────────────────────────────────────────────────
def recommend_properties(
    user_query: str,
    top_n: int = 5,
    verbose: bool = True
) -> tuple[pd.DataFrame, dict, str | None]:
    """
    End-to-end recommendation pipeline.

    Parameters
    ----------
    user_query : str
        Natural language query, e.g. "3BHK near hospital under 80 lakhs"
    top_n : int
        Number of top properties to return (default 5)
    verbose : bool
        Print progress to stdout

    Returns
    -------
    top_df : pd.DataFrame
        Top-N properties with original columns PLUS:
          predicted_price, value_score, final_score
        Sorted by final_score descending.
        Empty DataFrame if something fails.
    filters : dict
        Parsed filter dict from query_processor (useful for UI display).
    error : str | None
        Error or warning message, or None if all went smoothly.
    """
    # ── 1. Parse query ────────────────────────────────────────
    try:
        from query_processor import parse_query
        filters = parse_query(user_query)
        if verbose:
            print(f"\n  📝 Query: '{user_query}'")
            print(f"  🧠 Filters: {filters}")
    except Exception as e:
        return pd.DataFrame(), {}, f"Query parser error: {e}"

    # ── 2. Load resources ─────────────────────────────────────
    try:
        res = _load_resources()
        df_raw     = res["df"].copy()
        model      = res["model"]
        scaler     = res["scaler"]
        features   = res["features"]
    except FileNotFoundError as e:
        return pd.DataFrame(), filters, str(e)

    # ── 3. Apply filters ──────────────────────────────────────
    filtered = filter_properties(df_raw, filters)

    # ── 4. Fallback: relax filters if too few results ─────────
    warning_msg = None
    if len(filtered) < top_n:
        filtered, filters, warning_msg = _relax_filters(filters, filtered)

    if len(filtered) == 0:
        return pd.DataFrame(), filters, "No properties found even after relaxing filters. Try a broader query."

    # ── 5. Score and rank ─────────────────────────────────────
    try:
        scored = score_properties(filtered, model, scaler, features)
    except Exception as e:
        return pd.DataFrame(), filters, f"Scoring error: {e}"

    # ── 6. Sort and return top N ──────────────────────────────
    top_df = scored.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)

    if verbose:
        print(f"\n  🏆 Top {len(top_df)} recommendations:")
        print(f"  {'Rank':<5} {'Sector':<8} {'BHK':<5} {'Area':>8} {'Price':>10} {'Score':>8} {'Value':>8}")
        print("  " + "-"*58)
        for i, row in top_df.iterrows():
            print(
                f"  #{i+1:<4} S{row['Sector']:<7} {row['BHK']}BHK"
                f"  {row['Area_sqft']:>7.0f}  "
                f"  ₹{row['Price_Lakh']:>7.1f}L"
                f"  {row['final_score']:>8.3f}"
                f"  {row['value_score']:>+7.2f}"
            )

    return top_df, filters, warning_msg


# ─────────────────────────────────────────────────────────────
# HELPER — add Furnishing label back for display in app.py
# ─────────────────────────────────────────────────────────────
def get_furnishing_label(encoded_val) -> str:
    mapping = {0: "Unfurnished", 1: "Semi-Furnished", 2: "Furnished"}
    return mapping.get(int(encoded_val), "Unknown")


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PHASE 5 — Recommender Engine Test")
    print("=" * 60)

    test_queries = [
        "3BHK near hospital under 80 lakhs",
        "luxury villa in sector 7",
        "affordable 2bhk apartment semi furnished",
        "new construction near mall under 1.5 crore",
        "spacious 3bhk near park budget 60 to 120 lakhs",
    ]

    for query in test_queries:
        print(f"\n{'─'*60}")
        results, filters, err = recommend_properties(query, top_n=3)

        if err:
            print(f"  ⚠️  {err}")

        if not results.empty:
            print(f"\n  ✅ {len(results)} results found")
            for i, row in results.iterrows():
                print(f"\n  #{i+1}: Sector {row['Sector']} | {row['BHK']} BHK | "
                      f"₹{row['Price_Lakh']:.1f}L | {row['Area_sqft']:.0f} sqft")
                print(f"       Predicted: ₹{row['predicted_price']:.1f}L | "
                      f"Value score: {row['value_score']:+.2f} | "
                      f"Final score: {row['final_score']:.3f}")
        else:
            print("  ❌ No results")

    print("\n" + "=" * 60)
    print("✅ Phase 5 complete — recommender.py works!")
    print("   Next: Phase 6 — app.py (Streamlit UI)")
    print("=" * 60)