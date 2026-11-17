
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

from anomaly_engine import detect_anomalies


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="PriceIQ | Pricing Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = Path(__file__).resolve().parent
CSV_NAME = "customer_agreements.csv"


# ---------------------------------------------------------
# COLUMN NORMALIZATION
# ---------------------------------------------------------

ALIASES = {
    "region": [
        "region", "sales_region", "region_name"
    ],
    "district": [
        "district", "sales_district"
    ],
    "agreement_type": [
        "agreement_type", "agreement_type_code", "type"
    ],
    "agreement_number": [
        "agreement_number", "agreement_no", "agreement",
        "condition_record", "condition_record_number"
    ],
    "material": [
        "material", "material_number", "material_id",
        "product", "product_id"
    ],
    "customer": [
        "customer", "customer_id", "customer_number",
        "sold_to", "sold_to_party"
    ],
    "customer_name": [
        "customer_name", "name", "customer_description"
    ],
    "agreement_value": [
        "agreement_value", "value", "condition_value",
        "rate", "price", "amount"
    ],
    "unit": [
        "unit", "uom", "unit_of_measure",
        "sales_unit", "condition_unit"
    ],
    "valid_from": [
        "valid_from", "start_date", "validity_start",
        "from_date"
    ],
    "valid_to": [
        "valid_to", "end_date", "validity_end",
        "to_date", "expiry_date"
    ],
    "deleted": [
        "deleted", "deletion_flag", "delete_flag"
    ],
    "sales_rep": [
        "sales_rep", "employee", "sales_employee",
        "created_by"
    ],
    "condition_type": [
        "condition_type", "condition", "condition_code"
    ],
    "partner_function": [
        "partner_function", "partner_role"
    ]
}


def normalize_column_name(name):
    return (
        str(name).strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def standardize_columns(df):
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]

    rename_map = {}

    for canonical, candidates in ALIASES.items():
        if canonical in df.columns:
            continue

        for candidate in candidates:
            candidate = normalize_column_name(candidate)
            if candidate in df.columns:
                rename_map[candidate] = canonical
                break

    return df.rename(columns=rename_map)


# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------

def locate_customer_file():
    candidates = list(APP_DIR.rglob(CSV_NAME))

    if not candidates:
        return None

    # Prefer the expected data folder
    candidates.sort(
        key=lambda p: (
            0 if p.parent.name.lower() == "data" else 1,
            len(p.parts)
        )
    )

    return candidates[0]


@st.cache_data(show_spinner="Loading customer agreement data...")
def load_data(file_path):
    df = pd.read_csv(
        file_path,
        dtype=str,
        low_memory=False
    )

    df = standardize_columns(df)

    if "agreement_value" in df.columns:
        df["agreement_value"] = pd.to_numeric(
            df["agreement_value"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip(),
            errors="coerce"
        )

    for col in ["valid_from", "valid_to"]:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col], errors="coerce"
            )

    if "deleted" in df.columns:
        deleted = (
            df["deleted"].fillna("")
            .astype(str).str.strip().str.lower()
        )
        df["is_deleted"] = deleted.isin(
            ["x", "y", "yes", "true", "1"]
        )
    else:
        df["is_deleted"] = False

    return df


DATA_PATH = locate_customer_file()

if DATA_PATH is None:
    st.error(
        "PriceIQ customer data file was not found."
    )
    st.write("Expected file name:", CSV_NAME)
    st.write("Application directory:", str(APP_DIR))
    st.write("Current working directory:", str(Path.cwd()))
    st.write("Files and folders found:")

    for item in APP_DIR.iterdir():
        st.write(str(item))

    st.stop()


try:
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error("Unable to load the customer agreement file.")
    st.exception(exc)
    st.stop()


# ---------------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------------

active_df = df.loc[~df["is_deleted"]].copy()

if "agreement_value" not in active_df.columns:
    active_df["agreement_value"] = np.nan

if "material" not in active_df.columns:
    active_df["material"] = "Unknown"

if "customer" not in active_df.columns:
    active_df["customer"] = "Unknown"

if "unit" not in active_df.columns:
    active_df["unit"] = "Unknown"

if "customer_name" not in active_df.columns:
    active_df["customer_name"] = active_df["customer"]

if "region" not in active_df.columns:
    active_df["region"] = "Not available"

if "district" not in active_df.columns:
    active_df["district"] = "Not available"

if "valid_to" not in active_df.columns:
    active_df["valid_to"] = pd.NaT

if "valid_from" not in active_df.columns:
    active_df["valid_from"] = pd.NaT

if "agreement_type" not in active_df.columns:
    active_df["agreement_type"] = "Not available"

if "agreement_number" not in active_df.columns:
    active_df["agreement_number"] = "Not available"


# ---------------------------------------------------------
# RUN ANOMALY ENGINE
# ---------------------------------------------------------

@st.cache_data(show_spinner="Running AI pricing anomaly detection...")
def run_anomaly_detection(data):
    return detect_anomalies(
        data,
        value_col="agreement_value",
        material_col="material",
        unit_col="unit",
        customer_col="customer",
        date_from_col="valid_from",
        date_to_col="valid_to",
        contamination=0.03
    )


anomaly_df = run_anomaly_detection(active_df)


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------

st.sidebar.title("PriceIQ")
st.sidebar.caption("AI-powered Pricing Intelligence")

st.sidebar.success(
    f"{len(active_df):,} active agreement records loaded"
)

st.sidebar.caption(
    f"Data source: {DATA_PATH.relative_to(APP_DIR)}"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Agreement Explorer",
        "AI Anomaly Detection",
        "Pricing Intelligence",
        "Expiry & Renewals",
        "Price Benchmark",
        "Customer Profile",
        "Margin Guardrail Sandbox",
        "Data Readiness"
    ]
)


# ---------------------------------------------------------
# SHARED FILTERS
# ---------------------------------------------------------

def apply_filters(data):
    filtered = data.copy()

    st.sidebar.divider()
    st.sidebar.subheader("Data Filters")

    if "region" in filtered.columns:
        regions = sorted(
            filtered["region"].dropna().astype(str).unique()
        )
        selected_regions = st.sidebar.multiselect(
            "Region", regions, default=regions
        )
        filtered = filtered[
            filtered["region"].astype(str).isin(selected_regions)
        ]

    if "district" in filtered.columns:
        districts = sorted(
            filtered["district"].dropna().astype(str).unique()
        )
        selected_districts = st.sidebar.multiselect(
            "District", districts, default=districts
        )
        filtered = filtered[
            filtered["district"].astype(str).isin(selected_districts)
        ]

    return filtered


filtered_df = apply_filters(active_df)

filtered_anomalies = run_anomaly_detection(filtered_df)


def metric(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def show_table(data, columns=None):
    if columns:
        columns = [c for c in columns if c in data.columns]
        data = data[columns]

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )


def format_currency(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"


# ---------------------------------------------------------
# EXECUTIVE OVERVIEW
# ---------------------------------------------------------

if page == "Executive Overview":

    st.title("Executive Overview")
    st.caption(
        "A consolidated view of SAP pricing agreements and "
        "pricing review opportunities."
    )

    total_agreements = len(filtered_df)
    customers = filtered_df["customer"].nunique()
    materials = filtered_df["material"].nunique()
    flagged = int(filtered_anomalies["is_anomaly"].sum())

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Active Agreements", f"{total_agreements:,}")
    c2.metric("Customers", f"{customers:,}")
    c3.metric("Materials", f"{materials:,}")
    c4.metric("Records Flagged for Review", f"{flagged:,}")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Agreements by Region")
        region_counts = (
            filtered_df.groupby("region")
            .size().reset_index(name="Agreements")
        )
        st.bar_chart(
            region_counts.set_index("region")
        )

    with right:
        st.subheader("Agreement Type Distribution")
        type_counts = (
            filtered_df.groupby("agreement_type")
            .size().reset_index(name="Agreements")
        )
        st.bar_chart(
            type_counts.set_index("agreement_type")
        )

    st.subheader("Pricing Review Summary")

    if flagged:
        severity_counts = (
            filtered_anomalies[
                filtered_anomalies["is_anomaly"]
            ]["severity"]
            .value_counts()
            .rename_axis("Severity")
            .reset_index(name="Records")
        )
        st.bar_chart(
            severity_counts.set_index("Severity")
        )
    else:
        st.info("No anomalies were detected in the selected data.")

    st.caption(
        "Anomaly flags indicate records for investigation. "
        "They do not establish incorrect pricing or policy violations."
    )


# ---------------------------------------------------------
# AGREEMENT EXPLORER
# ---------------------------------------------------------

elif page == "Agreement Explorer":

    st.title("Agreement Explorer")
    st.caption("Search and inspect customer pricing agreements.")

    search = st.text_input(
        "Search customer, material, agreement or customer name"
    )

    result = filtered_df.copy()

    if search:
        search = search.lower()
        searchable = [
            c for c in [
                "customer", "customer_name", "material",
                "agreement_number", "agreement_type"
            ] if c in result.columns
        ]

        mask = pd.Series(False, index=result.index)

        for col in searchable:
            mask |= result[col].astype(str).str.lower().str.contains(
                search, na=False
            )

        result = result[mask]

    st.write(f"**{len(result):,} agreements found**")

    show_table(
        result,
        [
            "region", "district", "agreement_number",
            "agreement_type", "customer", "customer_name",
            "material", "agreement_value", "unit",
            "valid_from", "valid_to"
        ]
    )

    st.download_button(
        "Download filtered agreements",
        result.to_csv(index=False).encode("utf-8"),
        file_name="priceiq_filtered_agreements.csv",
        mime="text/csv"
    )


# ---------------------------------------------------------
# AI ANOMALY DETECTION
# ---------------------------------------------------------

elif page == "AI Anomaly Detection":

    st.title("AI Pricing Anomaly Detection")

    st.caption(
        "Machine learning and analytical checks identify unusual "
        "agreement patterns for commercial review."
    )

    st.info(
        "This module uses local machine learning and statistical "
        "analysis. No external AI API is required."
    )

    flagged_df = filtered_anomalies[
        filtered_anomalies["is_anomaly"]
    ].copy()

    total = len(filtered_anomalies)
    flagged = len(flagged_df)

    a, b, c, d = st.columns(4)

    a.metric("Records Analysed", f"{total:,}")
    b.metric("Flagged Records", f"{flagged:,}")
    c.metric(
        "High Priority",
        f"{(flagged_df['severity'] == 'High').sum():,}"
    )
    d.metric(
        "Flag Rate",
        f"{(flagged / total * 100) if total else 0:.1f}%"
    )

    st.divider()

    if flagged_df.empty:
        st.success("No anomalies were identified in this selection.")
    else:
        severity_filter = st.multiselect(
            "Filter by severity",
            ["High", "Medium"],
            default=["High", "Medium"]
        )

        type_values = sorted(
            set(
                "; ".join(
                    flagged_df["anomaly_type"].fillna("")
                ).split("; ")
            ) - {""}
        )

        type_filter = st.multiselect(
            "Filter by anomaly category",
            type_values,
            default=type_values
        )

        display = flagged_df[
            flagged_df["severity"].isin(severity_filter)
            & flagged_df["anomaly_type"].apply(
                lambda x: any(
                    t in str(x) for t in type_filter
                ) if type_filter else False
            )
        ]

        show_table(
            display,
            [
                "severity", "anomaly_type", "anomaly_reason",
                "customer", "customer_name", "material",
                "agreement_value", "unit",
                "relative_to_median_pct",
                "valid_from", "valid_to"
            ]
        )

        st.download_button(
            "Download anomaly review file",
            display.to_csv(index=False).encode("utf-8"),
            file_name="priceiq_anomaly_review.csv",
            mime="text/csv"
        )

    st.subheader("How the detection works")

    st.markdown("""
    - **Isolation Forest:** Identifies agreement values that are unusual
      relative to the overall dataset.
    - **Material price dispersion:** Compares agreement values for the
      same material and unit where enough comparison records exist.
    - **Duplicate patterns:** Finds repeated customer, material, unit
      and agreement-value combinations.
    - **Overlapping validity:** Identifies potentially overlapping
      validity periods for the same customer and material.

    A flagged record should be reviewed against the source SAP
    condition record and commercial context before taking action.
    """)


# ---------------------------------------------------------
# PRICING INTELLIGENCE
# ---------------------------------------------------------

elif page == "Pricing Intelligence":

    st.title("Pricing Intelligence")
    st.caption(
        "Explore agreement-value dispersion and investigate "
        "pricing patterns using the available SAP data."
    )

    materials = sorted(
        filtered_df["material"].dropna().astype(str).unique()
    )

    selected_material = st.selectbox(
        "Select material", materials
    )

    material_data = filtered_df[
        filtered_df["material"].astype(str) == selected_material
    ].copy()

    units = sorted(
        material_data["unit"].dropna().astype(str).unique()
    )

    selected_unit = st.selectbox("Select unit", units)

    material_data = material_data[
        material_data["unit"].astype(str) == selected_unit
    ]

    values = material_data["agreement_value"].dropna()

    x1, x2, x3, x4 = st.columns(4)

    x1.metric("Agreements", f"{len(material_data):,}")
    x2.metric(
        "Customers",
        f"{material_data['customer'].nunique():,}"
    )
    x3.metric(
        "Median Agreement Value",
        format_currency(values.median()) if len(values) else "N/A"
    )
    x4.metric(
        "Value Range",
        (
            f"{format_currency(values.min())} - "
            f"{format_currency(values.max())}"
        ) if len(values) else "N/A"
    )

    st.subheader("Agreement Value Distribution")

    if len(values):
        st.bar_chart(
            values.reset_index(drop=True).rename("Agreement Value")
        )

    st.subheader("Customer Agreement Comparison")

    comparison = (
        material_data.groupby(
            ["customer", "customer_name", "unit"],
            dropna=False
        )["agreement_value"]
        .agg(["count", "min", "median", "max"])
        .reset_index()
        .rename(columns={
            "count": "Agreement Count",
            "min": "Minimum Value",
            "median": "Median Value",
            "max": "Maximum Value"
        })
    )

    show_table(comparison)

    st.caption(
        "These are comparisons of maintained agreement values, "
        "not necessarily realized transaction prices."
    )


# ---------------------------------------------------------
# EXPIRY AND RENEWALS
# ---------------------------------------------------------

elif page == "Expiry & Renewals":

    st.title("Expiry & Renewals")

    today = pd.Timestamp.today().normalize()

    expiry_data = filtered_df.dropna(
        subset=["valid_to"]
    ).copy()

    expiry_data["days_to_expiry"] = (
        expiry_data["valid_to"] - today
    ).dt.days

    expired = expiry_data[
        expiry_data["days_to_expiry"] < 0
    ]

    next_30 = expiry_data[
        expiry_data["days_to_expiry"].between(0, 30)
    ]

    next_60 = expiry_data[
        expiry_data["days_to_expiry"].between(31, 60)
    ]

    next_90 = expiry_data[
        expiry_data["days_to_expiry"].between(61, 90)
    ]

    a, b, c, d = st.columns(4)

    a.metric("Expired", f"{len(expired):,}")
    b.metric("Expiring in 30 Days", f"{len(next_30):,}")
    c.metric("Expiring in 60 Days", f"{len(next_60):,}")
    d.metric("Expiring in 90 Days", f"{len(next_90):,}")

    st.subheader("Upcoming Expirations")

    upcoming = expiry_data[
        expiry_data["days_to_expiry"].between(0, 90)
    ].sort_values("valid_to")

    show_table(
        upcoming,
        [
            "customer", "customer_name", "material",
            "agreement_number", "agreement_value", "unit",
            "valid_to", "days_to_expiry"
        ]
    )

    st.download_button(
        "Download renewal list",
        upcoming.to_csv(index=False).encode("utf-8"),
        file_name="priceiq_renewal_list.csv",
        mime="text/csv"
    )


# ---------------------------------------------------------
# PRICE BENCHMARK
# ---------------------------------------------------------

elif page == "Price Benchmark":

    st.title("Price Benchmark")
    st.caption(
        "Compare maintained agreement values for the same "
        "material and unit."
    )

    materials = sorted(
        filtered_df["material"].dropna().astype(str).unique()
    )

    selected_material = st.selectbox(
        "Material", materials, key="benchmark_material"
    )

    material_data = filtered_df[
        filtered_df["material"].astype(str) == selected_material
    ]

    units = sorted(
        material_data["unit"].dropna().astype(str).unique()
    )

    selected_unit = st.selectbox(
        "Unit", units, key="benchmark_unit"
    )

    benchmark = material_data[
        material_data["unit"].astype(str) == selected_unit
    ].copy()

    benchmark["agreement_value"] = pd.to_numeric(
        benchmark["agreement_value"], errors="coerce"
    )

    benchmark = benchmark.dropna(
        subset=["agreement_value"]
    )

    if benchmark.empty:
        st.warning("No comparable agreement values are available.")
    else:
        median_value = benchmark["agreement_value"].median()

        benchmark["difference_from_median_pct"] = (
            benchmark["agreement_value"] / median_value - 1
        ) * 100 if median_value else np.nan

        st.metric(
            "Material / Unit Median",
            format_currency(median_value)
        )

        st.metric(
            "Agreement Records",
            f"{len(benchmark):,}"
        )

        show_table(
            benchmark.sort_values(
                "agreement_value", ascending=False
            ),
            [
                "customer", "customer_name", "material",
                "agreement_value", "unit",
                "difference_from_median_pct",
                "valid_from", "valid_to"
            ]
        )


# ---------------------------------------------------------
# CUSTOMER PROFILE
# ---------------------------------------------------------

elif page == "Customer Profile":

    st.title("Customer Profile")

    customer_options = (
        filtered_df[["customer", "customer_name"]]
        .drop_duplicates()
        .sort_values("customer_name")
    )

    customer_options["display"] = (
        customer_options["customer"].astype(str)
        + " | "
        + customer_options["customer_name"].astype(str)
    )

    selected = st.selectbox(
        "Select customer",
        customer_options["display"].tolist()
    )

    customer_id = selected.split(" | ")[0]

    customer_data = filtered_df[
        filtered_df["customer"].astype(str) == customer_id
    ].copy()

    a, b, c = st.columns(3)

    a.metric("Agreements", f"{len(customer_data):,}")
    b.metric(
        "Materials",
        f"{customer_data['material'].nunique():,}"
    )
    c.metric(
        "Agreement Value Median",
        format_currency(
            customer_data["agreement_value"].median()
        )
    )

    st.subheader("Customer Agreements")

    show_table(
        customer_data,
        [
            "agreement_number", "agreement_type",
            "material", "agreement_value", "unit",
            "valid_from", "valid_to", "region", "district"
        ]
    )


# ---------------------------------------------------------
# MARGIN GUARDRAIL SANDBOX
# ---------------------------------------------------------

elif page == "Margin Guardrail Sandbox":

    st.title("Margin Guardrail Sandbox")

    st.warning(
        "The current SAP agreement extracts do not contain verified "
        "product costs or realized sales prices. Enter assumptions "
        "below to test a hypothetical floor price."
    )

    col1, col2 = st.columns(2)

    with col1:
        cost = st.number_input(
            "Assumed product cost",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

        minimum_margin = st.slider(
            "Minimum gross margin (%)",
            min_value=0,
            max_value=90,
            value=20
        )

    with col2:
        list_price = st.number_input(
            "Assumed list price",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        maximum_discount = st.slider(
            "Maximum discount (%)",
            min_value=0,
            max_value=90,
            value=25
        )

    margin_floor = (
        cost / (1 - minimum_margin / 100)
        if minimum_margin < 100 else float("inf")
    )

    discount_floor = (
        list_price * (1 - maximum_discount / 100)
    )

    final_floor = max(margin_floor, discount_floor)

    st.divider()

    a, b, c = st.columns(3)

    a.metric("Margin-Based Floor", format_currency(margin_floor))
    b.metric("Discount-Based Floor", format_currency(discount_floor))
    c.metric("Minimum Permitted Price", format_currency(final_floor))

    st.caption(
        "This is an illustrative scenario based on user-entered "
        "assumptions. It is not an assessment of actual customer margin."
    )


# ---------------------------------------------------------
# DATA READINESS
# ---------------------------------------------------------

elif page == "Data Readiness":

    st.title("Data Readiness Assessment")

    st.caption(
        "Identify which additional data is needed to build "
        "advanced pricing intelligence."
    )

    required = {
        "Customer identifier": "customer",
        "Material identifier": "material",
        "Agreement value": "agreement_value",
        "Unit of measure": "unit",
        "Agreement start date": "valid_from",
        "Agreement end date": "valid_to",
        "Region": "region",
        "District": "district"
    }

    readiness_rows = []

    for label, col in required.items():
        present = col in df.columns
        non_null = (
            df[col].notna().sum()
            if present else 0
        )

        readiness_rows.append({
            "Data Field": label,
            "Available": "Yes" if present else "No",
            "Populated Records": f"{non_null:,}",
            "Coverage (%)": (
                round(non_null / len(df) * 100, 1)
                if present and len(df) else 0
            )
        })

    readiness = pd.DataFrame(readiness_rows)

    show_table(readiness)

    st.subheader("Additional Data Required")

    missing_data = pd.DataFrame([
        {
            "Data Requirement": "Invoice / billing history",
            "Purpose": "Realized selling price and sales trends"
        },
        {
            "Data Requirement": "Invoice quantities",
            "Purpose": "Demand modelling and price elasticity"
        },
        {
            "Data Requirement": "Product cost",
            "Purpose": "Actual gross margin and price floors"
        },
        {
            "Data Requirement": "Rebates and discounts",
            "Purpose": "Net price and pocket margin"
        },
        {
            "Data Requirement": "Freight and cost-to-serve",
            "Purpose": "Customer-level profitability"
        },
        {
            "Data Requirement": "Competitor prices",
            "Purpose": "Competitive price positioning"
        },
        {
            "Data Requirement": "Quote and win/loss history",
            "Purpose": "Quote conversion and price sensitivity"
        }
    ])

    show_table(missing_data)

    st.info(
        "The current agreement extracts support agreement visibility, "
        "benchmarking, expiry monitoring and anomaly review. They do "
        "not support validated demand elasticity, realized margin "
        "or revenue optimization."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.sidebar.divider()
st.sidebar.caption("PriceIQ 2.0 | Customer POC")
st.sidebar.caption("Local ML | No external AI API")
