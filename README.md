# PriceIQ Customer Agreement POC

This edition adapts PriceIQ to three customer-provided SAP pricing-agreement extracts:
- Region 1 / District 33 / A933
- Region 1 / District 33 / A963
- Region 4 / District 42 / A933

## What this version does
- Loads normalized customer agreement records
- Filters by region, district, extract and unit
- Searches by material, customer ID and customer name
- Shows agreement expiry and renewal worklists
- Compares maintained agreement values for the same material and unit
- Shows customer agreement profiles
- Provides a clearly labelled illustrative margin floor sandbox using user-entered cost and margin assumptions

## Important limitations
The supplied workbooks are agreement extracts. They do not contain invoice quantities, realized transaction prices, costs, rebates, freight or cost-to-serve. Therefore this version does **not** claim to calculate demand elasticity, actual gross/pocket margin, volume impact, revenue forecasts or optimized prices.

The app uses the normalized CSV generated from the three files. Do not put customer data in a public GitHub repository or an unapproved public hosting environment. Use a private repository and hosting approved by the customer / your organization. Confirm the customer's authorization before sharing or deploying the data.

## Run locally (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

## Files
- `app.py`: Streamlit application
- `data/customer_agreements.csv`: normalized agreement records
- `PriceIQ_Customer_Data_Assessment.xlsx`: profiling workbook (provided separately)
- `requirements.txt`: dependencies
