
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="PriceIQ | Customer Pricing Agreements", page_icon="💲", layout="wide")

DATA_PATH = Path("data/customer_agreements.csv")
AS_OF = pd.Timestamp.today().normalize()

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, dtype={"material": str, "customer_id": str, "district": str, "employee_no": str, "condition_id": str, "customer_name": str, "sales_rep": str}, low_memory=False)
    for c in ["valid_from", "valid_to"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["material", "customer_id", "customer_name", "unit", "sales_rep", "region", "district", "extract"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["agreement_value"] = pd.to_numeric(df["agreement_value"], errors="coerce")
    return df

df = load_data()

st.title("PriceIQ")
st.caption("Customer pricing intelligence | SAP agreement data | Customer-specific proof of concept")
st.info("This version is built from customer-provided SAP agreement extracts. It analyses agreement prices, coverage, expiry and price dispersion. It does not infer demand elasticity, realized margin or sales impact because transaction quantities, cost and realized net sales are not present in these files.")

with st.sidebar:
    st.header("Filter agreements")
    regions = sorted(df["region"].unique())
    selected_regions = st.multiselect("Region", regions, default=regions)
    districts = sorted(df[df["region"].isin(selected_regions)]["district"].unique())
    selected_districts = st.multiselect("District", districts, default=districts)
    extracts = sorted(df[df["district"].isin(selected_districts)]["extract"].unique())
    selected_extracts = st.multiselect("Extract", extracts, default=extracts)
    units = sorted(df["unit"].replace("", "Unknown").unique())
    selected_units = st.multiselect("Unit", units, default=units)
    include_expired = st.checkbox("Include expired agreements", value=True)
    search = st.text_input("Search material, customer ID or name")

f = df[
    df["region"].isin(selected_regions)
    & df["district"].isin(selected_districts)
    & df["extract"].isin(selected_extracts)
    & df["unit"].replace("", "Unknown").isin(selected_units)
].copy()
if not include_expired:
    f = f[f["valid_to"].isna() | (f["valid_to"] >= AS_OF)]
if search.strip():
    q = search.strip().lower()
    mask = (
        f["material"].str.lower().str.contains(q, na=False)
        | f["customer_id"].str.lower().str.contains(q, na=False)
        | f["customer_name"].str.lower().str.contains(q, na=False)
    )
    f = f[mask]

def money(v):
    return f"{v:,.2f}" if pd.notna(v) else "N/A"

total = len(f)
customers = f["customer_id"].nunique()
materials = f["material"].nunique()
expiry_60 = int(f["valid_to"].between(AS_OF, AS_OF + pd.Timedelta(days=60)).sum())
c1,c2,c3,c4 = st.columns(4)
c1.metric("Agreement records", f"{total:,}")
c2.metric("Customers", f"{customers:,}")
c3.metric("Materials", f"{materials:,}")
c4.metric("Expiring in 60 days", f"{expiry_60:,}")

tabs = st.tabs(["Executive Overview", "Agreement Explorer", "Expiry & Renewals", "Price Benchmark", "Customer Profile", "Guardrail Sandbox", "Data Readiness"])

with tabs[0]:
    st.subheader("Pricing agreement portfolio")
    left,right = st.columns(2)
    with left:
        expiry = f.copy()
        expiry["Expiry status"] = np.select(
            [expiry["valid_to"] < AS_OF,
             expiry["valid_to"].between(AS_OF, AS_OF + pd.Timedelta(days=30)),
             expiry["valid_to"].between(AS_OF + pd.Timedelta(days=31), AS_OF + pd.Timedelta(days=60)),
             expiry["valid_to"].between(AS_OF + pd.Timedelta(days=61), AS_OF + pd.Timedelta(days=90))],
            ["Expired","0-30 days","31-60 days","61-90 days"], default="Over 90 days"
        )
        exp_counts = expiry["Expiry status"].value_counts().reindex(["Expired","0-30 days","31-60 days","61-90 days","Over 90 days"], fill_value=0).rename_axis("Expiry status").reset_index(name="Agreements")
        st.plotly_chart(px.bar(exp_counts, x="Expiry status", y="Agreements", title="Agreement expiry exposure"), use_container_width=True)
    with right:
        unit_counts = f["unit"].replace("", "Unknown").value_counts().rename_axis("Unit").reset_index(name="Agreements")
        st.plotly_chart(px.pie(unit_counts, names="Unit", values="Agreements", title="Agreement mix by unit"), use_container_width=True)
    st.subheader("Agreement records by extract")
    by_extract = f.groupby(["region","district","extract"]).size().reset_index(name="Agreements")
    st.dataframe(by_extract, use_container_width=True, hide_index=True)
    st.caption("Agreement value is shown exactly as a numeric source value. The extracts do not include a currency code, so confirm currency with the SAP data owner before presenting monetary amounts. This is not necessarily a realized transaction price.")

with tabs[1]:
    st.subheader("Search and inspect agreements")
    cols = ["region","district","extract","material","customer_id","customer_name","agreement_value","unit","valid_from","valid_to","condition_id","sales_rep","employee_no"]
    st.dataframe(f[cols].sort_values(["valid_to","customer_name"], na_position="last"), use_container_width=True, hide_index=True, height=520)
    st.download_button("Download filtered agreements CSV", f[cols].to_csv(index=False).encode("utf-8"), "priceiq_filtered_agreements.csv", "text/csv")

with tabs[2]:
    st.subheader("Renewal and expiry worklist")
    days = st.slider("Show agreements expiring within next N days", 30, 240, 90, 15)
    due = f[f["valid_to"].between(AS_OF, AS_OF + pd.Timedelta(days=days))].copy()
    due["days_to_expiry"] = (due["valid_to"] - AS_OF).dt.days
    st.metric(f"Agreements expiring in {days} days", f"{len(due):,}")
    show = ["valid_to","days_to_expiry","customer_id","customer_name","material","agreement_value","unit","sales_rep","condition_id"]
    st.dataframe(due[show].sort_values(["valid_to","customer_name"]), use_container_width=True, hide_index=True, height=500)
    st.download_button("Download renewal worklist", due[show].to_csv(index=False).encode("utf-8"), "priceiq_renewal_worklist.csv", "text/csv")

with tabs[3]:
    st.subheader("Material and unit price benchmark")
    st.caption("Benchmarks compare maintained numeric agreement values for the same material and unit. The extracts do not include a currency code. These are descriptive, not recommendations; customer context, volumes, rebates and cost are not available here.")
    mats = sorted(f["material"].dropna().unique())
    chosen_material = st.selectbox("Material", mats)
    units_for_mat = sorted(f.loc[f.material.eq(chosen_material),"unit"].unique())
    chosen_unit = st.selectbox("Unit", units_for_mat)
    b = f[(f.material.eq(chosen_material)) & (f.unit.eq(chosen_unit))].copy()
    if len(b):
        med = b["agreement_value"].median()
        low = b["agreement_value"].min()
        high = b["agreement_value"].max()
        a,bm,c = st.columns(3)
        a.metric("Median agreement value", money(med))
        bm.metric("Lowest agreement value", money(low))
        c.metric("Highest agreement value", money(high))
        st.metric("Range vs median", f"{((high-low)/med*100):,.1f}%" if med else "N/A")
        st.plotly_chart(px.histogram(b, x="agreement_value", nbins=30, title=f"Agreement value distribution | {chosen_material} | {chosen_unit}"), use_container_width=True)
        st.dataframe(b[["customer_id","customer_name","agreement_value","valid_from","valid_to","sales_rep","condition_id"]].sort_values("agreement_value"), use_container_width=True, hide_index=True)
    else:
        st.warning("No agreement values found for this material and unit.")

with tabs[4]:
    st.subheader("Customer agreement profile")
    customer_options = f[["customer_id","customer_name"]].drop_duplicates().sort_values(["customer_name","customer_id"])
    if len(customer_options):
        labels = {f'{r.customer_name} | {r.customer_id}': r.customer_id for r in customer_options.itertuples()}
        choice = st.selectbox("Select customer", list(labels.keys()))
        cid = labels[choice]
        cdf = f[f.customer_id.eq(cid)].copy()
        x1,x2,x3 = st.columns(3)
        x1.metric("Agreements", f"{len(cdf):,}")
        x2.metric("Distinct materials", f"{cdf.material.nunique():,}")
        x3.metric("Sales representative", cdf.sales_rep.replace("", np.nan).dropna().mode().iat[0] if cdf.sales_rep.replace("", np.nan).notna().any() else "Not populated")
        st.dataframe(cdf[["material","agreement_value","unit","valid_from","valid_to","condition_id"]].sort_values(["material","unit"]), use_container_width=True, hide_index=True)
    else:
        st.warning("No customers match the selected filters.")

with tabs[5]:
    st.subheader("Illustrative margin guardrail sandbox")
    st.warning("Cost and minimum margin are not in the supplied agreement extracts. Enter them below to test a hypothetical floor. This is a user-supplied scenario, not a customer-approved policy.")
    mats = sorted(f["material"].dropna().unique())
    if mats:
        gm = st.selectbox("Choose material for scenario", mats, key="guard_material")
        gf = f[f.material.eq(gm)]
        gu = st.selectbox("Unit", sorted(gf.unit.unique()), key="guard_unit")
        gf = gf[gf.unit.eq(gu)]
        gc = gf[["customer_id","customer_name"]].drop_duplicates().sort_values("customer_name")
        glabels = {f"{r.customer_name} | {r.customer_id}":r.customer_id for r in gc.itertuples()}
        if glabels:
            gchoice = st.selectbox("Customer", list(glabels.keys()), key="guard_customer")
            selected = gf[gf.customer_id.eq(glabels[gchoice])]
            if len(selected):
                rec = selected.sort_values("valid_to").iloc[-1]
                st.write(f"Current agreement value: **{money(rec.agreement_value)} per {gu}** (currency code not supplied)")
                cost = st.number_input("Enter unit cost", min_value=0.0, value=0.0, step=1.0)
                min_margin = st.slider("Minimum gross margin %", 0, 80, 25)
                if cost > 0:
                    floor = cost / (1 - min_margin/100)
                    st.metric("Illustrative floor value", money(floor))
                    delta = rec.agreement_value - floor
                    if delta < 0:
                        st.error(f"Agreement value is {money(abs(delta))} below the illustrative floor value.")
                    else:
                        st.success(f"Agreement value is {money(delta)} above the illustrative floor value.")
                    st.caption("This check excludes rebates, freight, cost-to-serve, commissions, taxes and other pocket-price adjustments.")
    else:
        st.warning("No material available under current filters.")

with tabs[6]:
    st.subheader("What this data supports, and what is still missing")
    st.markdown("""
**Supported now**
- Customer and material agreement lookup
- Maintained agreement value comparison within the same material and unit
- Agreement expiry and renewal worklists
- Customer-specific agreement coverage
- Sales-representative and district slicing where populated
- Illustrative price-floor checks when an authorized user supplies cost and margin assumptions

**Not supported by these files alone**
- Price elasticity or demand response
- What-if volume, revenue or profit forecasts
- Actual gross margin or pocket margin
- Net price waterfall, rebates and cost-to-serve
- Automated optimal price recommendations

**Additional data required**
- Invoice / billing line history: customer, material, date, quantity, net realized price
- Cost: standard or actual cost at the same material / unit / date grain
- Discounts, rebates, freight, commissions and other deductions
- Optional but valuable: list price, quote / lost-quote data, competitor price, returns and order channel
- Agreement condition semantics and status/deletion semantics from SAP data dictionary or functional owner
    """)
    st.dataframe(pd.DataFrame([
        ["Customer/material agreement visibility","Available","High"],
        ["Agreement expiry management","Available","High"],
        ["Agreement price dispersion","Available","Medium"],
        ["Customer margin guardrails","Partial: needs cost + policy inputs","High"],
        ["Price elasticity","Not available: needs transactions","Critical"],
        ["What-if revenue / volume","Not available: needs transactions","Critical"],
        ["Pocket margin / price waterfall","Not available: needs rebates and cost-to-serve","High"],
        ["AI price recommendation","Not yet: needs validated demand + cost model","High"],
    ], columns=["Capability","Readiness from supplied files","Next dependency"]), use_container_width=True, hide_index=True)

st.divider()
st.caption("PriceIQ Customer POC | Built from the three supplied agreement extracts. Confirm SAP field meanings and customer authorization before external demonstration.")
