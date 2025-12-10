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

    required_1 = {"Recall Order", "Projected Order", "completed"}
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

    # File 1 basic stats
    total_file1_rows = len(df1)
    unique_file1_quotes = df1["quote_norm"].nunique(dropna=True)

    # Expected behavior from File 1
    expected_in_system = []
    expected_status = []
    for _, row in df1.iterrows():
        exp_in, exp_status = derive_expected_status(
            row["Projected Order"], row["completed"],
            won_label=won_label,
            noresp_label=noresp_label,
        )
        expected_in_system.append(exp_in)
        expected_status.append(exp_status)

    df1["expected_in_system"] = expected_in_system
    df1["expected_status"] = expected_status

    f1_expected = df1[df1["expected_in_system"]].copy()

    total_should_show_rows = len(f1_expected)
    unique_should_show = f1_expected["quote_norm"].nunique(dropna=True)

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
    unique_missing = merged.loc[~merged["found_in_system"], "quote_norm"].nunique(dropna=True)

    # Rows that should be present but are missing in File 2
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
    )

    # Mismatch analysis for Won
    won_mask = merged["expected_status"].str.casefold() == won_label.lower()
    won_found = merged[won_mask & merged["found_in_system"]].copy()
    wrong_won = won_found[
        won_found["Status"].fillna("").str.casefold() != won_label.lower()
    ].copy()

    # Mismatch analysis for No Response
    noresp_mask = merged["expected_status"].fillna("").str.casefold() == noresp_label.lower()
    noresp_found = merged[noresp_mask & merged["found_in_system"]].copy()
    wrong_noresp = noresp_found[
        noresp_found["Status"].fillna("").str.casefold() != noresp_label.lower()
    ].copy()

    summary = {
        # File 1
        "total_file1_rows": int(total_file1_rows),
        "unique_file1_quotes": int(unique_file1_quotes),
        # Expected presence
        "total_should_show_rows": int(total_should_show_rows),
        "unique_should_show": int(unique_should_show),
        # Actual presence
        "unique_found": int(unique_found),
        "unique_missing": int(unique_missing),
        "total_found_rows": int(total_found_rows),
        "total_missing_rows": int(total_missing_rows),
        # Status mismatches
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

    return summary, wrong_won_view, wrong_noresp_view, missing_view, merged


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
        "Upload File 1 (Recall Order, Projected Order, completed, etc.)",
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
                merged,
            ) = analyze_quotes(df1, df2, won_label=won_label, noresp_label=noresp_label)

            # ---- File 1 summary ----
            st.subheader("File 1 Summary")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Total uploaded rows (File 1)", summary["total_file1_rows"])
            with c2:
                st.metric(
                    "Unique quote number count (File 1)",
                    summary["unique_file1_quotes"],
                )

            st.markdown("---")

            # ---- Result summary (what should be in File 2) ----
            st.subheader("Result Summary (Quote presence in File 2)")

            c3, c4, c5 = st.columns(3)
            with c3:
                st.metric(
                    "Total UNIQUE quote numbers that SHOULD be in File 2",
                    summary["unique_should_show"],
                )
            with c4:
                st.metric(
                    "Available quote number count (unique)",
                    summary["unique_found"],
                )
            with c5:
                st.metric(
                    "Missing quote number count (unique)",
                    summary["unique_missing"],
                )

            st.markdown("---")

            # ---- Missing quote details + download ----
            st.subheader(
                "Quotes that SHOULD be in File 2 but are MISSING (unique list)"
            )
            if not missing_view.empty:
                st.dataframe(
                    missing_view.head(50),
                    use_container_width=True,
                    height=220,
                )

                # Download missing details
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
                st.info("No missing quotes based on the current logic.")

            st.markdown("---")

            # ---- Incorrect Won statuses ----
            st.subheader(f"Quotes that SHOULD be '{won_label}' but are different")
            st.write(
                f"Unique quotes with wrong status (expected '{won_label}'): "
                f"**{summary['wrong_won_count']}**"
            )
            if not wrong_won_view.empty:
                st.dataframe(
                    wrong_won_view.head(50),
                    use_container_width=True,
                    height=220,
                )
            else:
                st.info(f"No mismatches found for expected '{won_label}'.")

            st.markdown("---")

            # ---- Incorrect No Response statuses ----
            st.subheader(
                f"Quotes that SHOULD be '{noresp_label}' but are different"
            )
            st.write(
                f"Unique quotes with wrong status (expected '{noresp_label}'): "
                f"**{summary['wrong_noresp_count']}**"
            )
            if not wrong_noresp_view.empty:
                st.dataframe(
                    wrong_noresp_view.head(50),
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
            st.error(f"Error during processing: {e}")
