import os
import io
import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------
# BASIC PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="quote-status-mapper", layout="wide")

# Set to True only when debugging locally
DEBUG = False

# -------------------------------------------------
# AUTH CONFIG (SAFE FOR GITHUB)
# -------------------------------------------------


def get_valid_credentials():
    """
    Read credentials from Streamlit secrets first,
    then fall back to environment variables.
    Do NOT hard-code secrets in this file.
    """
    username = ""
    password = ""

    # From .streamlit/secrets.toml
    if "auth" in st.secrets:
        auth = st.secrets["auth"]
        username = auth.get("username", "") or username
        password = auth.get("password", "") or password

    # Fallback to environment variables (for local dev)
    username = os.getenv("APP_USERNAME", username)
    password = os.getenv("APP_PASSWORD", password)

    return username, password


VALID_USERNAME, VALID_PASSWORD = get_valid_credentials()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# -------------------------------------------------
# LOGIN GATE
# -------------------------------------------------
if not st.session_state.authenticated:
    st.title("quote-status-mapper – Login")

    username_input = st.text_input("Username")
    password_input = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if username_input == VALID_USERNAME and password_input == VALID_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.stop()

# -------------------------------------------------
# THRESHOLD CONFIG
# -------------------------------------------------
DEFAULT_THRESHOLD = 500  # dollars

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
# HELPERS
# -------------------------------------------------


def normalize_quote_id(value):
    """Normalize Recall Order / Quote Number for matching."""
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
    Data is only kept in memory; nothing is written to disk.
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
    Projected + completed -> expected status.

    Logic:
    - N / N : do not process
    - N / Y : Won
    - Y / Y : Won
    - Y / N : No Response
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
    Threshold logic:

    - Work only on quotes that are base_expected_in_system.
    - Sum Extended Price per quote_norm.
    - Get threshold by Taker Name (custom or default).
    - If total < threshold => quote is SCREENED (below_threshold=True).
    """

    if "Extended Price" not in df1.columns:
        raise ValueError("File 1 missing required column 'Extended Price' for threshold logic.")

    df1["_ext_price_num"] = pd.to_numeric(df1["Extended Price"], errors="coerce").fillna(0.0)

    df_base = df1[base_expected_mask].copy()

    if df_base.empty:
        df1["below_threshold"] = False
        screened_view = pd.DataFrame(
            columns=[
                "Quote ID (normalized)",
                "Recall Order (File 1)",
                "Taker Name",
                "Total Extended Price",
                "Threshold Limit",
                "Below Threshold?",
            ]
        )
        return df1, screened_view

    grp = (
        df_base.groupby("quote_norm", as_index=False)
        .agg({
            "_ext_price_num": "sum",
            "Taker Name": "first",
            "Recall Order": "first",
        })
        .rename(columns={"_ext_price_num": "total_extended_price"})
    )

    grp["taker_norm"] = grp["Taker Name"].astype(str).str.strip()
    grp["threshold_limit"] = grp["taker_norm"].map(CUSTOM_THRESHOLDS).fillna(DEFAULT_THRESHOLD)

    grp["below_threshold"] = grp["total_extended_price"] < grp["threshold_limit"]

    below_map = dict(zip(grp["quote_norm"], grp["below_threshold"]))
    df1["below_threshold"] = df1["quote_norm"].map(below_map).fillna(False)

    screened_quotes = grp[grp["below_threshold"]].copy()

    screened_view = screened_quotes[[
        "quote_norm",
        "Recall Order",
        "Taker Name",
        "total_extended_price",
        "threshold_limit",
        "below_threshold",
    ]].rename(columns={
        "quote_norm": "Quote ID (normalized)",
        "Recall Order": "Recall Order (File 1)",
        "Taker Name": "Taker Name",
        "total_extended_price": "Total Extended Price",
        "threshold_limit": "Threshold Limit",
        "below_threshold": "Below Threshold?",
    }).sort_values("Total Extended Price")

    return df1, screened_view


def analyze_quotes(df1, df2, won_label, noresp_label):
    """
    Main comparison:
    - apply status logic
    - apply threshold screening
    - compare to File 2
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

    df1["quote_norm"] = df1["Recall Order"].apply(normalize_quote_id)
    df2["quote_norm"] = df2["Quote Number"].apply(normalize_quote_id)

    total_file1_rows = len(df1)
    unique_file1_quotes = df1["quote_norm"].nunique(dropna=True)

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

    df1, screened_view = apply_threshold_screening(df1, base_expected_mask)

    df1["expected_in_system"] = df1["base_expected_in_system"] & (~df1["below_threshold"])
    f1_expected = df1[df1["expected_in_system"]].copy()

    total_should_show_rows = len(f1_expected)
    unique_should_show = f1_expected["quote_norm"].nunique(dropna=True)

    screened_unique_count = (
        screened_view["Quote ID (normalized)"].nunique(dropna=True)
        if not screened_view.empty else 0
    )

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
    unique_missing = merged.loc[~merged["found_in_system"], "quote_norm"].nunique(dropna=True)

    missing_view = merged.loc[~merged["found_in_system"]].copy()
    missing_view = (
        missing_view[["Recall Order", "quote_norm", "expected_status",
                      "Projected Order", "completed"]]
        .drop_duplicates()
        .rename(columns={
            "Recall Order": "Recall Order (File 1)",
            "quote_norm": "Quote ID (normalized)",
            "expected_status": "Expected Status",
            "Projected Order": "Projected Order (File 1)",
            "completed": "Completed (File 1)",
        })
        .sort_values("Quote ID (normalized)")
    )

    won_mask = merged["expected_status"].str.casefold() == won_label.lower()
    won_found = merged[won_mask & merged["found_in_system"]].copy()
    wrong_won = won_found[
        won_found["Status"].fillna("").str.casefold() != won_label.lower()
    ].copy()

    noresp_mask = merged["expected_status"].fillna("").str.casefold() == noresp_label.lower()
    noresp_found = merged[noresp_mask & merged["found_in_system"]].copy()
    wrong_noresp = noresp_found[
        noresp_found["Status"].fillna("").str.casefold() != noresp_label.lower()
    ].copy()

    summary = {
        "total_file1_rows": int(total_file1_rows),
        "unique_file1_quotes": int(unique_file1_quotes),
        "screened_unique_count": int(screened_unique_count),
        "total_should_show_rows": int(total_should_show_rows),
        "unique_should_show": int(unique_should_show),
        "unique_found": int(unique_found),
        "unique_missing": int(unique_missing),
        "total_found_rows": int(total_found_rows),
        "total_missing_rows": int(total_missing_rows),
        "wrong_won_count": int(wrong_won["quote_norm"].nunique(dropna=True)),
        "wrong_noresp_count": int(wrong_noresp["quote_norm"].nunique(dropna=True)),
    }

    wrong_won_view = (
        wrong_won[["Recall Order", "expected_status", "Status"]]
        .drop_duplicates()
        .rename(
            columns={
                "Recall Order": "Recall Order (File 1)",
                "Status": "Actual Status (File 2)",
                "expected_status": "Expected Status",
            }
        )
    )

    wrong_noresp_view = (
        wrong_noresp[["Recall Order", "expected_status", "Status"]]
        .drop_duplicates()
        .rename(
            columns={
                "Recall Order": "Recall Order (File 1)",
                "Status": "Actual Status (File 2)",
                "expected_status": "Expected Status",
            }
        )
    )

    return summary, wrong_won_view, wrong_noresp_view, missing_view, screened_view, merged


# -------------------------------------------------
# UI
# -------------------------------------------------

st.title("quote-status-mapper")

with st.expander("Status Logic Settings (optional)", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        won_label = st.text_input("Text of 'Won' status in File 2", value="Won")
    with col_b:
        noresp_label = st.text_input(
            "Text of 'No Response' status in File 2", value="No Response"
        )

col1, col2 = st.columns(2)

with col1:
    st.subheader("File 1 – Recall / Order Data")
    file1 = st.file_uploader(
        "Upload File 1 (Recall Order, Projected Order, completed, Extended Price, etc.)",
        type=["xlsx", "xls", "csv"],
        key="file1",
    )

with col2:
    st.subheader("File 2 – Quote System Export")
    file2 = st.file_uploader(
        "Upload File 2 (Quote Number, Status, ...)",
        type=["xlsx", "xls", "csv"],
        key="file2",
    )

st.markdown("---")

if st.button("Run Comparison"):
    if file1 is None or file2 is None:
        st.error("Please upload BOTH File 1 and File 2.")
    else:
        try:
            df1 = read_any_table(file1)
            df2 = read_any_table(file2)

            (
                summary,
                wrong_won_view,
                wrong_noresp_view,
                missing_view,
                screened_view,
                merged,
            ) = analyze_quotes(df1, df2, won_label=won_label, noresp_label=noresp_label)

            # File 1 summary
            st.subheader("File 1 Summary")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total uploaded rows (File 1)", summary["total_file1_rows"])
            with c2:
                st.metric(
                    "Unique quote number count (File 1)",
                    summary["unique_file1_quotes"],
                )
            with c3:
                st.metric(
                    "Screened quotes below threshold (unique)",
                    summary["screened_unique_count"],
                )

            st.markdown("---")

            # Result summary after threshold
            st.subheader("Result Summary (quotes that should be in File 2 AFTER threshold)")

            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric(
                    "Total UNIQUE quote numbers that SHOULD be in File 2",
                    summary["unique_should_show"],
                )
            with r2:
                st.metric(
                    "Available quote number count (unique)",
                    summary["unique_found"],
                )
            with r3:
                st.metric(
                    "Missing quote number count (unique)",
                    summary["unique_missing"],
                )

            st.markdown("---")

            # Screened quotes
            st.subheader("Screened Quotes (below threshold – not processed)")
            if not screened_view.empty:
                st.write(
                    "These quotes met the Projected/Completed logic but were "
                    "screened out because their total Extended Price is below "
                    "the taker's threshold."
                )
                st.dataframe(
                    screened_view,
                    use_container_width=True,
                    height=260,
                )

                buf_screen = io.BytesIO()
                with pd.ExcelWriter(buf_screen, engine="openpyxl") as writer:
                    screened_view.to_excel(
                        writer, index=False, sheet_name="ScreenedQuotes"
                    )
                buf_screen.seek(0)
                st.download_button(
                    label="Download Screened Quotes (with totals & thresholds)",
                    data=buf_screen,
                    file_name="screened_quotes.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                )
            else:
                st.info("No quotes were screened out by threshold logic.")

            st.markdown("---")

            # Missing quotes
            st.subheader("Quotes that SHOULD be in File 2 but are MISSING (unique list)")
            if not missing_view.empty:
                st.dataframe(
                    missing_view,
                    use_container_width=True,
                    height=260,
                )

                miss_buf = io.BytesIO()
                with pd.ExcelWriter(miss_buf, engine="openpyxl") as writer:
                    missing_view.to_excel(
                        writer, index=False, sheet_name="MissingQuotes"
                    )
                miss_buf.seek(0)

                st.download_button(
                    label="Download Missing Quote Details",
                    data=miss_buf,
                    file_name="missing_quotes.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                )
            else:
                st.info("No missing quotes (after threshold).")

            st.markdown("---")

            # Wrong won
            st.subheader(f"Quotes that SHOULD be '{won_label}' but are different")
            st.write(
                f"Unique quotes with wrong status (expected '{won_label}'): "
                f"**{summary['wrong_won_count']}**"
            )
            if not wrong_won_view.empty:
                st.dataframe(
                    wrong_won_view,
                    use_container_width=True,
                    height=220,
                )
            else:
                st.info(f"No mismatches found for expected '{won_label}'.")

            st.markdown("---")

            # Wrong no-response
            st.subheader(f"Quotes that SHOULD be '{noresp_label}' but are different")
            st.write(
                f"Unique quotes with wrong status (expected '{noresp_label}'): "
                f"**{summary['wrong_noresp_count']}**"
            )
            if not wrong_noresp_view.empty:
                st.dataframe(
                    wrong_noresp_view,
                    use_container_width=True,
                    height=220,
                )
            else:
                st.info(f"No mismatches found for expected '{noresp_label}'.")

            st.markdown("---")
            st.subheader("Download Full Detailed Comparison (optional)")

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                merged.to_excel(writer, index=False, sheet_name="Comparison")
            buffer.seek(0)

            st.download_button(
                label="Download Detailed Comparison Excel",
                data=buffer,
                file_name="quote_status_comparison.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
            )

        except Exception as e:
            if DEBUG:
                st.exception(e)
            else:
                st.error(
                    "An unexpected error occurred while processing the files. "
                    "Please confirm that both files have the required columns "
                    "and that numeric values (like Extended Price) are valid."
                )
