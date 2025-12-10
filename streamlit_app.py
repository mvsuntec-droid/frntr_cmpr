import os
import io
import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------
# BASIC PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="quote-status-mapper", layout="wide")

# Set to True only during local testing
DEBUG = False

# -------------------------------------------------
# SECURE AUTH (SAFE FOR GITHUB + STREAMLIT CLOUD)
# -------------------------------------------------


def get_valid_credentials():
    """
    Load credentials in this order:
    1. Streamlit Cloud secrets
    2. Environment variables
    3. Fallback default (matt / Interlynx123)
       — remove this once real secrets are set
    """
    username = None
    password = None

    # 1) From Streamlit secrets on deployment
    if "auth" in st.secrets:
        auth = st.secrets["auth"]
        username = auth.get("username", username)
        password = auth.get("password", password)

    # 2) Environment variables (local dev or server)
    if username is None:
        username = os.getenv("APP_USERNAME")
    if password is None:
        password = os.getenv("APP_PASSWORD")

    # 3) Safe fallback for first-time GitHub usage
    if username is None or password is None:
        username = "matt"
        password = "Interlynx123"

    return username, password


VALID_USERNAME, VALID_PASSWORD = get_valid_credentials()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# -------------------------------------------------
# LOGIN PAGE
# -------------------------------------------------
if not st.session_state.authenticated:
    st.title("quote-status-mapper — Login Required")

    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")

    if st.button("Login"):
        if username_input == VALID_USERNAME and password_input == VALID_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.stop()

# -------------------------------------------------
# THRESHOLD CONFIGURATION
# -------------------------------------------------
DEFAULT_THRESHOLD = 500

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

CUSTOM_THRESHOLDS = {k.strip(): v for k, v in CUSTOM_THRESHOLDS.items()}

# -------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------


def normalize_quote_id(value):
    """Convert Recall Order / Quote Number into a clean comparable value."""
    if pd.isna(value):
        return None
    s = str(value).strip().replace(",", "")
    try:
        return str(int(float(s)))
    except Exception:
        return s


def read_any_table(uploaded_file):
    """Supports CSV and Excel with safe encoding fallbacks."""
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

    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file)


def derive_expected_status(projected_val, completed_val, won, noresp):
    """Core logic based on Projected Order + completed."""
    def yn(val):
        if val is None:
            return "N"
        return "Y" if str(val).strip().upper().startswith("Y") else "N"

    p, c = yn(projected_val), yn(completed_val)

    if p == "N" and c == "N":
        return False, None
    if (p == "N" and c == "Y") or (p == "Y" and c == "Y"):
        return True, won
    if p == "Y" and c == "N":
        return True, noresp

    return False, None


def apply_threshold_screening(df1, base_mask):
    """Apply custom threshold logic per Taker Name."""
    if "Extended Price" not in df1.columns:
        raise ValueError("Missing Extended Price column.")

    df1["_ext_num"] = pd.to_numeric(df1["Extended Price"], errors="coerce").fillna(0.0)

    df_base = df1[base_mask].copy()
    if df_base.empty:
        df1["below_threshold"] = False
        return df1, pd.DataFrame()

    grp = (
        df_base.groupby("quote_norm", as_index=False)
        .agg({
            "_ext_num": "sum",
            "Taker Name": "first",
            "Recall Order": "first",
        })
        .rename(columns={"_ext_num": "total_extended_price"})
    )

    grp["taker_norm"] = grp["Taker Name"].astype(str).str.strip()
    grp["threshold_limit"] = grp["taker_norm"].map(CUSTOM_THRESHOLDS).fillna(DEFAULT_THRESHOLD)

    grp["below_threshold"] = grp["total_extended_price"] < grp["threshold_limit"]

    map_thresh = dict(zip(grp["quote_norm"], grp["below_threshold"]))
    df1["below_threshold"] = df1["quote_norm"].map(map_thresh).fillna(False)

    screened = grp[grp["below_threshold"]].copy()

    screened_view = screened.rename(columns={
        "quote_norm": "Quote ID (normalized)",
        "Recall Order": "Recall Order (File 1)",
    })

    return df1, screened_view


def analyze_quotes(df1, df2, won_label, noresp_label):
    """Main processing function."""

    df1.columns = [c.strip() for c in df1.columns]
    df2.columns = [c.strip() for c in df2.columns]

    required_1 = {
        "Recall Order", "Projected Order", "completed", "Taker Name", "Extended Price"
    }
    required_2 = {"Quote Number", "Status"}

    if not required_1.issubset(df1.columns):
        raise ValueError(f"File 1 missing: {required_1 - set(df1.columns)}")

    if not required_2.issubset(df2.columns):
        raise ValueError(f"File 2 missing: {required_2 - set(df2.columns)}")

    df1["quote_norm"] = df1["Recall Order"].apply(normalize_quote_id)
    df2["quote_norm"] = df2["Quote Number"].apply(normalize_quote_id)

    # Initial expectations (pre-threshold)
    df1["base_expected_in_system"] = False
    df1["expected_status"] = None

    for idx, row in df1.iterrows():
        exp_in, exp_stat = derive_expected_status(
            row["Projected Order"], row["completed"], won_label, noresp_label
        )
        df1.at[idx, "base_expected_in_system"] = exp_in
        df1.at[idx, "expected_status"] = exp_stat

    # Apply threshold logic
    df1, screened_view = apply_threshold_screening(df1, df1["base_expected_in_system"])

    df1["expected_in_system"] = df1["base_expected_in_system"] & (~df1["below_threshold"])

    # Filter records actually expected
    expected_df = df1[df1["expected_in_system"]].copy()

    # Merge with File 2 statuses
    merged = expected_df.merge(
        df2[["quote_norm", "Status"]],
        on="quote_norm",
        how="left"
    )

    merged["found_in_system"] = merged["Status"].notna()

    # Missing quotes
    missing = merged[~merged["found_in_system"]].copy()

    # Summary stats
    summary = {
        "total_file1_rows": len(df1),
        "unique_file1_quotes": df1["quote_norm"].nunique(dropna=True),
        "screened_unique_count": screened_view["Quote ID (normalized)"].nunique()
        if not screened_view.empty else 0,
        "unique_should_show": expected_df["quote_norm"].nunique(),
        "unique_found": merged.loc[merged["found_in_system"], "quote_norm"].nunique(),
        "unique_missing": missing["quote_norm"].nunique(),
    }

    return summary, missing, screened_view, merged


# -------------------------------------------------
# UI
# -------------------------------------------------

st.title("quote-status-mapper")

file_col1, file_col2 = st.columns(2)

with file_col1:
    st.subheader("Upload File 1 (Recall / Order Data)")
    file1 = st.file_uploader(
        "Accepted: CSV, XLS, XLSX", type=["csv", "xls", "xlsx"], key="file1"
    )

with file_col2:
    st.subheader("Upload File 2 (Quote Data)")
    file2 = st.file_uploader(
        "Accepted: CSV, XLS, XLSX", type=["csv", "xls", "xlsx"], key="file2"
    )

st.markdown("---")

if st.button("Run Comparison"):
    if not file1 or not file2:
        st.error("Please upload both files.")
        st.stop()

    try:
        df1 = read_any_table(file1)
        df2 = read_any_table(file2)

        summary, missing_view, screened_view, merged = analyze_quotes(
            df1, df2, won_label="Won", noresp_label="No Response"
        )

        # -------------------------
        # SUMMARY
        # -------------------------
        st.subheader("Summary")
        colA, colB, colC = st.columns(3)

        with colA:
            st.metric("Total Rows (File 1)", summary["total_file1_rows"])
            st.metric("Screened Quotes (Below Threshold)", summary["screened_unique_count"])

        with colB:
            st.metric("Unique Quotes in File 1", summary["unique_file1_quotes"])
            st.metric("Quotes Expected in File 2", summary["unique_should_show"])

        with colC:
            st.metric("Found in File 2", summary["unique_found"])
            st.metric("Missing in File 2", summary["unique_missing"])

        st.markdown("---")

        # -------------------------
        # SCREENED QUOTES
        # -------------------------
        st.subheader("Screened Quotes (Below Threshold — Not Processed)")

        if screened_view.empty:
            st.info("No quotes were screened out by threshold logic.")
        else:
            st.dataframe(screened_view, use_container_width=True, height=260)

            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as wr:
                screened_view.to_excel(wr, index=False, sheet_name="ScreenedQuotes")
            buf.seek(0)

            st.download_button(
                "Download Screened Quotes",
                data=buf,
                file_name="screened_quotes.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.markdown("---")

        # -------------------------
        # MISSING QUOTES
        # -------------------------
        st.subheader("Quotes EXPECTED but NOT FOUND in File 2")

        if summary["unique_missing"] == 0:
            st.success("All expected quotes were found in File 2.")
        else:
            st.dataframe(missing_view, use_container_width=True, height=260)

            buf2 = io.BytesIO()
            with pd.ExcelWriter(buf2, engine="openpyxl") as wr:
                missing_view.to_excel(wr, index=False, sheet_name="MissingQuotes")
            buf2.seek(0)

            st.download_button(
                "Download Missing Quotes",
                data=buf2,
                file_name="missing_quotes.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        st.markdown("---")

        # -------------------------
        # FULL COMPARISON DOWNLOAD
        # -------------------------
        st.subheader("Full Comparison Export")

        buf3 = io.BytesIO()
        with pd.ExcelWriter(buf3, engine="openpyxl") as wr:
            merged.to_excel(wr, index=False, sheet_name="Comparison")
        buf3.seek(0)

        st.download_button(
            "Download Full Comparison",
            data=buf3,
            file_name="quote_status_comparison.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        if DEBUG:
            st.exception(e)
        else:
            st.error("An unexpected error occurred. Please verify your files and try again.")
