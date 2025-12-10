import streamlit as st
import pandas as pd
import numpy as np
import io

# -------------------------------------------------
# BASIC PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="quote-status-mapper", layout="wide")

# -------------------------------------------------
# SIMPLE AUTH (USERNAME / PASSWORD)
# -------------------------------------------------
VALID_USERNAME = "matt"
VALID_PASSWORD = "Interlynx123"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("quote-status-mapper – Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.stop()

# -------------------------------------------------
# THRESHOLD CONFIG
# -------------------------------------------------
DEFAULT_THRESHOLD = 500  # dollars

# Custom thresholds per taker name (must match "Taker Name" in File 1)
CUSTOM_THRESHOLDS = {
    "Tony Phillips": 2400,
    "Brian Johnson": 2000,
    "Spencer Vivlamore": 2000,
    "Kyran Wilson": 1800,
    "Barry Brandon": 1500,
    "Jerry Cartwright": 1500,
    "Daniel Sajo": 1500,
    "Tia Valencia": 1500,
    "Phil Jentzen": 1200,
    "Mary Jane Gutierrez": 1200,
    "Mark A Roy": 1100,
    "Alyssa Agnes": 500,
    "Dave Hoeffel": 500,
    "Kristina Jackson": 500,
    "Vicki McKenzie": 500,
    "Timothy Motis": 500,
    "Amillia Bohlken": 500,
    "Barry Kniffen": 500,
    "Ashley Smither": 500,
    "Marita Antolin": 500,
    "Darien Fejarang": 500,
    "Amy Fernandez": 500,
    "Jon McDonald": 500,
    "Jason Miclat": 500,
    "Jason Munoz": 500,
    "Jason Quiambao": 500,
    "Chris Fox": 500,
    "Bill Vivlamore": 500,
    "Roudel Nucum": 500,
    "Rodney Glen": 500,
    "FAI - Job Quotes Team": 500,
    "ANC - Job Quotes Team": 500,
    "GUM - Job Quotes Team": 500,
    "Steve Lopez": 500,
    "Mark Generoso": 500,
    "Catalina Wilson": 500,
    "Patrick St Peter": 500,
}
# Normalize keys once
CUSTOM_THRESHOLDS = {k.strip(): v for k, v in CUSTOM_THRESHOLDS.items()}

# -------------------------------------------------
# HELPERS (NO GLOBAL DATA STATE)
# -------------------------------------------------


def normalize_quote_id(value):
    """Normalize Recall Order / Quote Number."""
    if pd.isna(value):
        return None
    s = str(value).strip().replace(",", "")
    try:
        return str(int(float(s)))
    except Exception:
        return s


def read_any_table(uploaded_file):
    """
    Read CSV or Excel into a DataFrame.
    CSVs: try utf-8, then latin-1, then latin-1 with errors='ignore'.
    """
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        for enc in ("utf-8", "latin-1"):
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=enc)
            except UnicodeDecodeError:
                continue
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1", errors="ignore")
    else:
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)


def derive_expected_status(projected_val, completed_val, won_label, noresp_label):
    """
    Map Projected Order + completed to expected behavior.

    Rules:

    Projected   completed   expected_in_system   expected_status
    ------------------------------------------------------------
    N           N           False               None
    N           Y           True                won_label
    Y           Y           True                won_label
    Y           N           True                noresp_label
    """
    def yn(x):
        if x is None:
            return "N"
        s = str(x).strip().upper()
        return "Y" if s.startswith("Y") else "N"

    p = yn(projected_val)
    c = yn(completed_val)

    if p == "N" and c == "N":
        return False, None
    if (p == "N" and c == "Y") or (p == "Y" and c == "Y"):
        return True, won_label
    if p == "Y" and c == "N":
        return True, noresp_label

    return False, None


def apply_threshold_screening(df1, base_expected_mask):
    """
    Apply threshold rules to quotes that are expected to be in the system.

    - Sum Extended Price by quote (Recall Order / quote_norm)
    - Use Taker Name to pick a threshold (custom or default)
    - Quotes whose TOTAL < threshold are "screened" and will NOT be processed.
    """

    # Ensure numeric Extended Price
    ext_col = "Extended Price"
    if ext_col not in df1.columns:
        raise ValueError(f"File 1 missing required column '{ext_col}' for threshold logic.")

    # Make a numeric copy
    df1["_ext_price_num"] = pd.to_numeric(df1[ext_col], errors="coerce").fillna(0.0)

    # Work only on rows that were expected (before threshold)
    df_base = df1[base_expected_mask].copy()

    if df_base.empty:
        # Nothing to screen
        df1["below_threshold"] = False
        screened_view = pd.DataFrame(
            columns=["Quote ID (normalized)", "Recall Order (File 1)",
                     "Taker Name", "Total Extended Price", "Threshold Limit"]
        )
        return df1, screened_view

    # Aggregate by quote
    grp = (
        df_base.groupby("quote_norm", as_index=False)
        .agg({
            "_ext_price_num": "sum",
            "Taker Name": "first",
            "Recall Order": "first",
        })
        .rename(columns={"_ext_price_num": "total_extended_price"})
    )

    # Apply thresholds by taker
    grp["taker_norm"] = grp["Taker Name"].astype(str).str.strip()
    grp["threshold_limit"] = grp["taker_norm"].map(CUSTOM_THRESHOLDS).fillna(DEFAULT_THRESHOLD)

    grp["below_threshold"] = grp["total_extended_price"] < grp["threshold_limit"]

    # Screened quotes (unique)
    screened_quotes = grp[grp["below_threshold"]].copy()

    # Map below_threshold flag back to df1 rows
    below_map = dict(zip(grp["quote_norm"], grp["below_threshold"]))
    df1["below_threshold"] = df1["quote_norm"].map(below_map).fillna(False)

    # Build screened view for display/download
    screened_view = screened_quotes[[
        "quote_norm", "Recall Order", "Taker Name",
        "total_extended_price", "threshold_limit"
    ]].rename(columns={
        "quote_norm": "Quote ID (normalized)",
        "Recall Order": "Recall Order (File 1)",
        "Taker Name": "Taker Name",
        "total_extended_price": "Total Extended Price",
        "threshold_limit": "Threshold Limit",
    })

    return df1, screened_view


def analyze_quotes(df1, df2, won_label, noresp_label):
    """
    Core comparison logic.
    df1: Recall/Order data
    df2: Quote system export
    """

    df1 = df1.copy()
    df2 = df2.copy()

    df1.columns = [c.strip() for c in df1.columns]
    df2.columns = [c.strip() for c in df2.columns]

    required_1 = {"Recall Order", "Projected Order", "completed", "Taker Name", "Extended Price"}
    required_2 = {"Quote Number", "Status"}

    if not required_1.issubset(df1.columns):
        missing = required_1 - set(df1.columns)
        raise ValueError(f"File 1 missing required column(s): {missing}")
    if not required_2.issubset(df2.columns):
        missing = required_2 - set(df2.columns)
        raise ValueError(f"File 2 missing required column(s): {missing}")

    # Normalize quote IDs
    df1["quote_norm"] = df1["Recall Order"].apply(normalize_quote_id)
    df2["quote_norm"] = df2["Quote Number"].apply(normalize_quote_id)

    # File 1 base stats
    total_file1_rows = len(df1)
    unique_file1_quotes = df1["quote_norm"].nunique(dropna=True)

    # Expected behavior from File 1 BEFORE threshold
    base_expected_in_system = []
    expected_status = []
    for _, row in df1.iterrows():
        exp_in, exp_status = derive_expected_status(
            row["Projected Order"], row["completed"],
            won_label=won_label,
            noresp_label=noresp_label,
        )
        base_expected_in_system.append(exp_in)
        expected_status.append(exp_status)

    df1["base_expected_in_system"] = base_expected_in_system
    df1["expected_status"] = expected_status

    base_expected_mask = df1["base_expected_in_system"]

    # ---- Threshold screening (per quote / taker) ----
    df1, screened_view = apply_threshold_screening(df1, base_expected_mask)

    # Final "expected_in_system" after threshold screening
    df1["expected_in_system"] = df1["base_expected_in_system"] & (~df1["below_threshold"])

    # Quotes that should be in system AFTER threshold
    f1_expected = df1[df1["expected_in_system"]].copy()

    total_should_show_rows = len(f1_expected)
    unique_should_show = f1_expected["quote_norm"].nunique(dropna=True)

    # Screened quotes count (unique)
    screened_unique_count = screened_view["Quote ID (normalized)"].nunique(
        dropna=True
    ) if not screened_view.empty else 0

    # File 2 minimal just for matching
    df2_min = df2[["quote_norm", "Status"]].copy()

    merged = f1_expected.merge(
        df2_min,
        on="quote_norm",
        how="left",
        suffixes=("", "_sys"),
    )

    merged["found_in_system"] = merged["Status"].notna()

    total_found_rows = int(merged["found_in_system"].sum())
    total_missing_rows = int(total_should_show_rows - total_found_rows)

    unique_found = merged.loc[merged["found_in_system"], "quote_norm"].nunique(dropna=True)
    unique_missing = merged.loc[~merged["found_in_system"], "quote_norm"].nunique(dropna=T_]()_
