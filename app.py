import os
import json
import math
import hmac
import sqlite3
from io import BytesIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# =========================================================
# CONFIG
# =========================================================
load_dotenv()

APP_PASSWORD = os.getenv("APP_PASSWORD", "EDDAQAQ2026")
DB_FILE = "clinic_forecast.db"

st.set_page_config(
    page_title="EDDAQAQ EXP - Prévisionnel Clinique",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #071120 0%, #0c1a33 100%);
        color: white;
    }
    .block-container {
        max-width: 1550px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 16px;
        min-height: 110px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }
    .metric-title {
        font-size: 0.92rem;
        opacity: 0.82;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-sub {
        font-size: 0.82rem;
        opacity: 0.72;
        margin-top: 6px;
    }
    .section-box {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# AUTH
# =========================================================
def check_password():
    if st.session_state.get("authenticated", False):
        return True

    st.title("EDDAQAQ EXP - Prévisionnel Clinique")
    st.info("Accès sécurisé")

    pwd = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter", use_container_width=True):
        if hmac.compare_digest(pwd, APP_PASSWORD):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect")
    st.stop()

check_password()

# =========================================================
# HELPERS
# =========================================================
def metric_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def style_plot(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white")),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
    return fig

def fmt_money(x):
    try:
        return f"{float(x):,.2f}".replace(",", " ")
    except Exception:
        return "-"

def safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        if isinstance(x, str):
            x = x.replace(" ", "").replace(",", ".")
        return float(x)
    except Exception:
        return default

def json_dumps_df(df: pd.DataFrame) -> str:
    return df.to_json(orient="records", force_ascii=False)

def json_loads_df(payload: str, columns: list):
    if not payload:
        return pd.DataFrame(columns=columns)
    try:
        df = pd.read_json(payload)
        for c in columns:
            if c not in df.columns:
                df[c] = np.nan
        return df[columns]
    except Exception:
        return pd.DataFrame(columns=columns)

# =========================================================
# DB
# =========================================================
def get_db():
    con = sqlite3.connect(DB_FILE, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    return con

def init_db():
    con = get_db()
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_state (
            section TEXT PRIMARY KEY,
            payload TEXT,
            updated_at TEXT
        )
        """
    )
    con.commit()
    con.close()

init_db()

def save_section(section: str, payload: str):
    con = get_db()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO app_state(section, payload, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(section) DO UPDATE SET
            payload=excluded.payload,
            updated_at=excluded.updated_at
        """,
        (section, payload, datetime.now().isoformat(timespec="seconds"))
    )
    con.commit()
    con.close()

def load_section(section: str, default_payload: str):
    con = get_db()
    cur = con.cursor()
    cur.execute("SELECT payload FROM app_state WHERE section = ?", (section,))
    row = cur.fetchone()
    con.close()
    return row[0] if row and row[0] else default_payload

# =========================================================
# DEFAULT DATA
# =========================================================
def get_default_settings():
    return {
        "project_name": "Clinique Multidisciplinaire",
        "company_name": "Société CLINIQUE CAMILIA",
        "currency": "MAD",
        "start_year": 2025,
        "nb_years": 5,
        "months_year1": 12,
        "opening_cash": 0.0,
        "inflation_pct": 2.0,
        "corp_tax_pct": 20.0,
        "solidarity_tax_pct": 1.5,
        "local_tax_y1": 0.0,
        "local_tax_growth_pct": 2.0,
        "dividend_y1": 0.0,
        "dividend_y2": 0.0,
        "dividend_y3": 0.0,
        "dividend_y4": 0.0,
        "dividend_y5": 0.0,
    }

REVENUE_COLS = [
    "Activité", "Type",
    "Jours/an", "Tarif", "Part clinique %",
    "Base Y1", "Croissance Y2 %", "Croissance Y3 %", "Croissance Y4 %", "Croissance Y5 %"
]

FIXED_COLS = [
    "Libellé", "Montant Y1", "Croissance annuelle %"
]

VARIABLE_COLS = [
    "Libellé", "Mode", "Valeur Y1", "Croissance annuelle %"
]

PAYROLL_COLS = [
    "Poste", "Type",
    "Effectif Y1", "Effectif Y2", "Effectif Y3", "Effectif Y4", "Effectif Y5",
    "Salaire mensuel net Y1", "Croissance salaire %", "Charges patronales %"
]

CAPEX_COLS = [
    "Actif", "Année achat", "Montant",
    "Durée amortissement", "Mode financement",
    "Taux financé %", "Taux intérêt %", "Durée financement",
    "Premier loyer / apport initial"
]

BFR_COLS = [
    "Hypothèse", "Y1", "Y2", "Y3", "Y4", "Y5"
]

def default_revenue_df():
    return pd.DataFrame([
        ["Radiologie", "actes/jour", 300, 2500, 70, 2.0, 50, 20, 10, 5],
        ["Hospitalisation", "lits*taux_occ", 365, 700, 50, 0.20, 10, 10, 10, 10],
        ["Consultations", "actes/jour", 300, 300, 20, 12.0, 15, 10, 8, 8],
        ["Pharmacie", "forfait_annuel", 0, 0, 100, 1000000, 15, 10, 8, 6],
    ], columns=REVENUE_COLS)

def default_fixed_df():
    return pd.DataFrame([
        ["Loyer clinique", 250000 * 12, 0],
        ["Redevance crédit-bail", 669969.40, 8],
        ["Assurances", 120000, 3],
        ["Maintenance matériel", 180000, 4],
        ["Honoraires", 120000, 3],
        ["Déchets médicaux", 72000, 3],
        ["Télécom", 36000, 2],
        ["Publicité", 100000, 5],
        ["Frais bancaires", 24000, 2],
        ["Nettoyage / sécurité", 180000, 3],
    ], columns=FIXED_COLS)

def default_variable_df():
    return pd.DataFrame([
        ["Médicaments", "%CA", 8.0, 0],
        ["Consommables médicaux", "%CA", 4.5, 0],
        ["Pharmacie - achats", "%CA", 18.0, 0],
        ["Fournitures de bureau", "Montant", 80000, 3],
    ], columns=VARIABLE_COLS)

def default_payroll_df():
    return pd.DataFrame([
        ["Infirmiers", "Employé", 10, 12, 14, 15, 16, 7000, 4, 21],
        ["Médecins salariés", "Employé", 4, 5, 6, 6, 7, 22000, 4, 21],
        ["Personnel administratif", "Employé", 4, 5, 5, 6, 6, 8000, 3, 21],
        ["Dirigeant", "Dirigeant", 1, 1, 1, 1, 1, 25000, 3, 0],
    ], columns=PAYROLL_COLS)

def default_capex_df():
    return pd.DataFrame([
        ["IRM", 1, 11040913, 7, "Leasing", 100, 7, 7, 0],
        ["Scanner", 1, 4832710, 7, "Leasing", 100, 7, 7, 0],
        ["Aménagement clinique", 1, 4974648, 10, "Cash", 0, 0, 0, 0],
    ], columns=CAPEX_COLS)

def default_bfr_df():
    return pd.DataFrame([
        ["Part CA AMO %", 80, 80, 80, 80, 80],
        ["Part CA assurances privées %", 10, 10, 10, 10, 10],
        ["Part CA sans couverture %", 10, 10, 10, 10, 10],
        ["Stock en jours d'achats", 15, 15, 15, 15, 15],
        ["Créances AMO en jours", 120, 120, 120, 120, 120],
        ["Créances privées en jours", 120, 120, 120, 120, 120],
        ["Créances sans couverture en jours", 0, 0, 0, 0, 0],
        ["Autres créances en jours CA", 30, 30, 30, 30, 30],
        ["Dettes fournisseurs en jours achats", 60, 60, 60, 60, 60],
        ["Autres dettes en jours CA", 30, 30, 30, 30, 30],
    ], columns=BFR_COLS)

# =========================================================
# LOAD STATE
# =========================================================
def load_all_state():
    settings = json.loads(load_section("settings", json.dumps(get_default_settings(), ensure_ascii=False)))
    revenue_df = json_loads_df(load_section("revenue", json_dumps_df(default_revenue_df())), REVENUE_COLS)
    fixed_df = json_loads_df(load_section("fixed_costs", json_dumps_df(default_fixed_df())), FIXED_COLS)
    variable_df = json_loads_df(load_section("variable_costs", json_dumps_df(default_variable_df())), VARIABLE_COLS)
    payroll_df = json_loads_df(load_section("payroll", json_dumps_df(default_payroll_df())), PAYROLL_COLS)
    capex_df = json_loads_df(load_section("capex", json_dumps_df(default_capex_df())), CAPEX_COLS)
    bfr_df = json_loads_df(load_section("bfr", json_dumps_df(default_bfr_df())), BFR_COLS)
    return settings, revenue_df, fixed_df, variable_df, payroll_df, capex_df, bfr_df

settings, revenue_df, fixed_df, variable_df, payroll_df, capex_df, bfr_df = load_all_state()

# =========================================================
# CALCULATIONS
# =========================================================
def get_years(settings):
    start_year = int(safe_float(settings.get("start_year", 2025), 2025))
    nb_years = int(safe_float(settings.get("nb_years", 5), 5))
    return [start_year + i for i in range(nb_years)]

def get_growth_factor(year_index, row, growth_cols):
    if year_index == 0:
        return 1.0
    growth = 1.0
    for i in range(1, year_index + 1):
        pct = safe_float(row[growth_cols[i - 1]], 0.0)
        growth *= (1 + pct / 100.0)
    return growth

def build_revenue_forecast(settings, revenue_df):
    years = get_years(settings)
    months_year1 = max(1, min(12, int(safe_float(settings.get("months_year1", 12), 12))))

    rows = []
    total_by_year = {y: 0.0 for y in years}

    for _, row in revenue_df.iterrows():
        activity = str(row.get("Activité", "")).strip() or "Activité"
        typ = str(row.get("Type", "actes/jour")).strip()
        days_open = safe_float(row.get("Jours/an"), 0)
        price = safe_float(row.get("Tarif"), 0)
        clinic_share = safe_float(row.get("Part clinique %"), 100) / 100.0
        base_y1 = safe_float(row.get("Base Y1"), 0)

        yearly_values = {}
        for i, y in enumerate(years):
            factor = get_growth_factor(
                i,
                row,
                ["Croissance Y2 %", "Croissance Y3 %", "Croissance Y4 %", "Croissance Y5 %"]
            )

            if typ == "actes/jour":
                amount = days_open * base_y1 * factor * price * clinic_share
            elif typ == "lits*taux_occ":
                # Base Y1 = taux d'occupation (ex: 0.20 pour 20%)
                amount = days_open * base_y1 * factor * price * clinic_share
            elif typ == "forfait_annuel":
                amount = base_y1 * factor * clinic_share
            else:
                amount = base_y1 * factor

            if i == 0:
                amount *= months_year1 / 12.0

            yearly_values[y] = amount
            total_by_year[y] += amount

        out_row = {"Activité": activity, "Type": typ}
        out_row.update(yearly_values)
        rows.append(out_row)

    df = pd.DataFrame(rows)
    total_row = {"Activité": "TOTAL", "Type": ""}
    total_row.update(total_by_year)
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    return df

def build_fixed_costs_forecast(settings, fixed_df):
    years = get_years(settings)
    months_year1 = max(1, min(12, int(safe_float(settings.get("months_year1", 12), 12))))
    rows = []
    totals = {y: 0.0 for y in years}

    for _, row in fixed_df.iterrows():
        label = str(row.get("Libellé", "")).strip() or "Charge fixe"
        base = safe_float(row.get("Montant Y1"), 0)
        growth = safe_float(row.get("Croissance annuelle %"), 0)

        vals = {}
        for i, y in enumerate(years):
            amount = base * ((1 + growth / 100.0) ** i)
            if i == 0:
                amount *= months_year1 / 12.0
            vals[y] = amount
            totals[y] += amount

        r = {"Libellé": label}
        r.update(vals)
        rows.append(r)

    df = pd.DataFrame(rows)
    total_row = {"Libellé": "TOTAL"}
    total_row.update(totals)
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    return df

def build_variable_costs_forecast(settings, variable_df, revenue_total_map):
    years = get_years(settings)
    months_year1 = max(1, min(12, int(safe_float(settings.get("months_year1", 12), 12))))
    rows = []
    totals = {y: 0.0 for y in years}

    for _, row in variable_df.iterrows():
        label = str(row.get("Libellé", "")).strip() or "Charge variable"
        mode = str(row.get("Mode", "%CA")).strip()
        value_y1 = safe_float(row.get("Valeur Y1"), 0)
        growth = safe_float(row.get("Croissance annuelle %"), 0)

        vals = {}
        for i, y in enumerate(years):
            if mode == "%CA":
                amount = revenue_total_map[y] * (value_y1 / 100.0)
            else:
                amount = value_y1 * ((1 + growth / 100.0) ** i)
                if i == 0:
                    amount *= months_year1 / 12.0

            vals[y] = amount
            totals[y] += amount

        r = {"Libellé": label, "Mode": mode}
        r.update(vals)
        rows.append(r)

    df = pd.DataFrame(rows)
    total_row = {"Libellé": "TOTAL", "Mode": ""}
    total_row.update(totals)
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    return df

def build_payroll_forecast(settings, payroll_df):
    years = get_years(settings)
    months_year1 = max(1, min(12, int(safe_float(settings.get("months_year1", 12), 12))))

    rows = []
    salary_totals = {y: 0.0 for y in years}
    employer_totals = {y: 0.0 for y in years}
    total_totals = {y: 0.0 for y in years}

    eff_cols = ["Effectif Y1", "Effectif Y2", "Effectif Y3", "Effectif Y4", "Effectif Y5"]

    for _, row in payroll_df.iterrows():
        position = str(row.get("Poste", "")).strip() or "Poste"
        ptype = str(row.get("Type", "Employé")).strip()
        salary_monthly_y1 = safe_float(row.get("Salaire mensuel net Y1"), 0)
        salary_growth = safe_float(row.get("Croissance salaire %"), 0)
        charges_pct = safe_float(row.get("Charges patronales %"), 0) / 100.0

        vals_salary = {}
        vals_charges = {}
        vals_total = {}

        for i, y in enumerate(years):
            eff = safe_float(row.get(eff_cols[i]), 0)
            monthly_salary = salary_monthly_y1 * ((1 + salary_growth / 100.0) ** i)
            annual_salary = eff * monthly_salary * 12
            if i == 0:
                annual_salary *= months_year1 / 12.0

            employer = annual_salary * charges_pct if ptype.lower() != "dirigeant" else 0.0
            total = annual_salary + employer

            vals_salary[y] = annual_salary
            vals_charges[y] = employer
            vals_total[y] = total

            salary_totals[y] += annual_salary
            employer_totals[y] += employer
            total_totals[y] += total

        out_row = {"Poste": position, "Type": ptype}
        for y in years:
            out_row[f"Salaires {y}"] = vals_salary[y]
            out_row[f"Charges {y}"] = vals_charges[y]
            out_row[f"Total {y}"] = vals_total[y]
        rows.append(out_row)

    df = pd.DataFrame(rows)

    return {
        "detail": df,
        "salary_totals": salary_totals,
        "employer_totals": employer_totals,
        "total_totals": total_totals,
    }

def annuity_payment(principal, annual_rate_pct, years):
    principal = safe_float(principal, 0)
    annual_rate = safe_float(annual_rate_pct, 0) / 100.0
    years = int(safe_float(years, 0))
    if principal <= 0 or years <= 0:
        return 0.0
    if annual_rate == 0:
        return principal / years
    return principal * (annual_rate / (1 - (1 + annual_rate) ** (-years)))

def build_capex_and_financing(settings, capex_df):
    years = get_years(settings)
    capex_by_year = {y: 0.0 for y in years}
    depreciation_by_year = {y: 0.0 for y in years}
    loan_inflows_by_year = {y: 0.0 for y in years}
    loan_principal_by_year = {y: 0.0 for y in years}
    loan_interest_by_year = {y: 0.0 for y in years}
    lease_payment_by_year = {y: 0.0 for y in years}
    upfront_by_year = {y: 0.0 for y in years}

    schedule_rows = []

    for _, row in capex_df.iterrows():
        asset = str(row.get("Actif", "")).strip() or "Actif"
        buy_year_index = max(1, int(safe_float(row.get("Année achat"), 1)))
        if buy_year_index > len(years):
            continue

        buy_year = years[buy_year_index - 1]
        amount = safe_float(row.get("Montant"), 0)
        dep_life = max(1, int(safe_float(row.get("Durée amortissement"), 1)))
        mode = str(row.get("Mode financement", "Cash")).strip()
        financed_pct = safe_float(row.get("Taux financé %"), 0) / 100.0
        rate = safe_float(row.get("Taux intérêt %"), 0)
        fin_years = max(0, int(safe_float(row.get("Durée financement"), 0)))
        upfront = safe_float(row.get("Premier loyer / apport initial"), 0)

        capex_by_year[buy_year] += amount
        upfront_by_year[buy_year] += upfront

        annual_dep = amount / dep_life
        start_idx = buy_year_index - 1
        for i in range(start_idx, min(start_idx + dep_life, len(years))):
            depreciation_by_year[years[i]] += annual_dep

        financed_amount = amount * financed_pct

        if mode.lower() == "loan":
            loan_inflows_by_year[buy_year] += financed_amount
            payment = annuity_payment(financed_amount, rate, fin_years)
            balance = financed_amount

            for i in range(start_idx, min(start_idx + fin_years, len(years))):
                y = years[i]
                interest = balance * (rate / 100.0)
                principal = max(0.0, payment - interest)
                if principal > balance:
                    principal = balance
                    payment = principal + interest
                loan_interest_by_year[y] += interest
                loan_principal_by_year[y] += principal
                balance -= principal

        elif mode.lower() == "leasing":
            payment = annuity_payment(financed_amount, rate, fin_years)
            for i in range(start_idx, min(start_idx + fin_years, len(years))):
                y = years[i]
                lease_payment_by_year[y] += payment

        schedule_rows.append({
            "Actif": asset,
            "Année achat": buy_year,
            "Montant": amount,
            "Mode": mode,
            "Amortissement annuel": annual_dep,
            "Premier loyer / apport": upfront,
        })

    schedule_df = pd.DataFrame(schedule_rows)

    return {
        "schedule_df": schedule_df,
        "capex_by_year": capex_by_year,
        "depreciation_by_year": depreciation_by_year,
        "loan_inflows_by_year": loan_inflows_by_year,
        "loan_principal_by_year": loan_principal_by_year,
        "loan_interest_by_year": loan_interest_by_year,
        "lease_payment_by_year": lease_payment_by_year,
        "upfront_by_year": upfront_by_year,
    }

def bfr_lookup(bfr_df, label):
    row = bfr_df[bfr_df["Hypothèse"] == label]
    if row.empty:
        return [0, 0, 0, 0, 0]
    return [
        safe_float(row.iloc[0]["Y1"], 0),
        safe_float(row.iloc[0]["Y2"], 0),
        safe_float(row.iloc[0]["Y3"], 0),
        safe_float(row.iloc[0]["Y4"], 0),
        safe_float(row.iloc[0]["Y5"], 0),
    ]

def build_bfr_forecast(settings, bfr_df, revenue_total_map, variable_total_map):
    years = get_years(settings)

    amo_share = bfr_lookup(bfr_df, "Part CA AMO %")
    priv_share = bfr_lookup(bfr_df, "Part CA assurances privées %")
    no_cover_share = bfr_lookup(bfr_df, "Part CA sans couverture %")
    stock_days = bfr_lookup(bfr_df, "Stock en jours d'achats")
    amo_days = bfr_lookup(bfr_df, "Créances AMO en jours")
    priv_days = bfr_lookup(bfr_df, "Créances privées en jours")
    no_cover_days = bfr_lookup(bfr_df, "Créances sans couverture en jours")
    other_rec_days = bfr_lookup(bfr_df, "Autres créances en jours CA")
    supplier_days = bfr_lookup(bfr_df, "Dettes fournisseurs en jours achats")
    other_debt_days = bfr_lookup(bfr_df, "Autres dettes en jours CA")

    rows = []
    bfr_map = {}
    delta_map = {}

    prev_bfr = 0.0
    for i, y in enumerate(years):
        revenue = revenue_total_map[y]
        purchases = variable_total_map[y]

        stock = purchases / 360 * stock_days[i]
        rec_amo = (revenue * amo_share[i] / 100.0) / 360 * amo_days[i]
        rec_priv = (revenue * priv_share[i] / 100.0) / 360 * priv_days[i]
        rec_no = (revenue * no_cover_share[i] / 100.0) / 360 * no_cover_days[i]
        other_rec = revenue / 360 * other_rec_days[i]
        debt_sup = purchases / 360 * supplier_days[i]
        other_debt = revenue / 360 * other_debt_days[i]

        bfr = stock + rec_amo + rec_priv + rec_no + other_rec - debt_sup - other_debt
        delta = bfr - prev_bfr

        bfr_map[y] = bfr
        delta_map[y] = delta
        prev_bfr = bfr

        rows.append({
            "Année": y,
            "Stock": stock,
            "Créances AMO": rec_amo,
            "Créances privées": rec_priv,
            "Créances sans couverture": rec_no,
            "Autres créances": other_rec,
            "Dettes fournisseurs": debt_sup,
            "Autres dettes": other_debt,
            "BFR": bfr,
            "Variation BFR": delta,
        })

    df = pd.DataFrame(rows)
    return df, bfr_map, delta_map

def build_pnl_forecast(settings, revenue_total_map, variable_total_map, fixed_total_map, payroll_total_map, capex_fin):
    years = get_years(settings)

    corp_tax = safe_float(settings.get("corp_tax_pct", 20), 20) / 100.0
    solidarity = safe_float(settings.get("solidarity_tax_pct", 1.5), 1.5) / 100.0
    local_tax_y1 = safe_float(settings.get("local_tax_y1", 0), 0)
    local_tax_growth = safe_float(settings.get("local_tax_growth_pct", 2), 2)

    rows = []
    net_result_map = {}
    ebitda_map = {}

    for i, y in enumerate(years):
        revenue = revenue_total_map[y]
        variable_costs = variable_total_map[y]
        gross_margin = revenue - variable_costs
        fixed_costs = fixed_total_map[y]
        payroll = payroll_total_map[y]
        local_taxes = local_tax_y1 * ((1 + local_tax_growth / 100.0) ** i)
        depreciation = capex_fin["depreciation_by_year"][y]
        interest = capex_fin["loan_interest_by_year"][y]

        ebitda = gross_margin - fixed_costs - payroll - local_taxes
        ebit = ebitda - depreciation
        ebt = ebit - interest

        is_tax = max(0.0, ebt) * corp_tax
        solidarity_tax = max(0.0, ebt) * solidarity
        net_result = ebt - is_tax - solidarity_tax

        ebitda_map[y] = ebitda
        net_result_map[y] = net_result

        rows.append({
            "Année": y,
            "Chiffre d'affaires": revenue,
            "Charges variables": variable_costs,
            "Marge brute": gross_margin,
            "Charges fixes": fixed_costs,
            "Charges de personnel": payroll,
            "Impôts et taxes d'exploitation": local_taxes,
            "EBITDA": ebitda,
            "Amortissements": depreciation,
            "EBIT": ebit,
            "Charges financières": interest,
            "Résultat avant impôt": ebt,
            "IS": is_tax,
            "Taxe solidarité": solidarity_tax,
            "Résultat net": net_result,
        })

    return pd.DataFrame(rows), ebitda_map, net_result_map

def build_cashflow_forecast(settings, pnl_df, delta_bfr_map, capex_fin):
    years = get_years(settings)
    opening_cash = safe_float(settings.get("opening_cash", 0), 0)

    dividends = {
        years[0]: safe_float(settings.get("dividend_y1", 0), 0),
        years[1]: safe_float(settings.get("dividend_y2", 0), 0) if len(years) > 1 else 0,
        years[2]: safe_float(settings.get("dividend_y3", 0), 0) if len(years) > 2 else 0,
        years[3]: safe_float(settings.get("dividend_y4", 0), 0) if len(years) > 3 else 0,
        years[4]: safe_float(settings.get("dividend_y5", 0), 0) if len(years) > 4 else 0,
    }

    rows = []
    cash_open = opening_cash

    pnl_map = pnl_df.set_index("Année").to_dict("index")

    for y in years:
        ebitda = pnl_map[y]["EBITDA"]
        is_tax = pnl_map[y]["IS"]
        solidarity = pnl_map[y]["Taxe solidarité"]
        delta_bfr = delta_bfr_map[y]

        cfo = ebitda - is_tax - solidarity - delta_bfr

        capex = capex_fin["capex_by_year"][y]
        upfront = capex_fin["upfront_by_year"][y]
        cfi = -capex - upfront

        loan_inflow = capex_fin["loan_inflows_by_year"][y]
        loan_principal = capex_fin["loan_principal_by_year"][y]
        lease_payment = capex_fin["lease_payment_by_year"][y]
        dividends_paid = dividends.get(y, 0)

        cff = loan_inflow - loan_principal - lease_payment - dividends_paid

        net_cf = cfo + cfi + cff
        cash_close = cash_open + net_cf

        rows.append({
            "Année": y,
            "EBITDA": ebitda,
            "IS": is_tax,
            "Taxe solidarité": solidarity,
            "Variation BFR": delta_bfr,
            "Cash flow exploitation": cfo,
            "CAPEX": capex,
            "Premier loyer / apport initial": upfront,
            "Cash flow investissement": cfi,
            "Entrées emprunts": loan_inflow,
            "Remboursement capital": loan_principal,
            "Paiements leasing": lease_payment,
            "Dividendes": dividends_paid,
            "Cash flow financement": cff,
            "Variation nette trésorerie": net_cf,
            "Trésorerie début": cash_open,
            "Trésorerie fin": cash_close,
        })

        cash_open = cash_close

    return pd.DataFrame(rows)

def run_full_model(settings, revenue_df, fixed_df, variable_df, payroll_df, capex_df, bfr_df):
    revenue_forecast = build_revenue_forecast(settings, revenue_df)
    years = get_years(settings)
    revenue_total_map = revenue_forecast[revenue_forecast["Activité"] == "TOTAL"][years].iloc[0].to_dict()

    fixed_forecast = build_fixed_costs_forecast(settings, fixed_df)
    fixed_total_map = fixed_forecast[fixed_forecast["Libellé"] == "TOTAL"][years].iloc[0].to_dict()

    variable_forecast = build_variable_costs_forecast(settings, variable_df, revenue_total_map)
    variable_total_map = variable_forecast[variable_forecast["Libellé"] == "TOTAL"][years].iloc[0].to_dict()

    payroll_forecast = build_payroll_forecast(settings, payroll_df)
    capex_fin = build_capex_and_financing(settings, capex_df)
    bfr_forecast, bfr_map, delta_bfr_map = build_bfr_forecast(settings, bfr_df, revenue_total_map, variable_total_map)

    pnl_df, ebitda_map, net_result_map = build_pnl_forecast(
        settings,
        revenue_total_map,
        variable_total_map,
        fixed_total_map,
        payroll_forecast["total_totals"],
        capex_fin
    )

    cashflow_df = build_cashflow_forecast(settings, pnl_df, delta_bfr_map, capex_fin)

    return {
        "years": years,
        "revenue_forecast": revenue_forecast,
        "fixed_forecast": fixed_forecast,
        "variable_forecast": variable_forecast,
        "payroll_forecast": payroll_forecast,
        "capex_fin": capex_fin,
        "bfr_forecast": bfr_forecast,
        "pnl_df": pnl_df,
        "cashflow_df": cashflow_df,
        "revenue_total_map": revenue_total_map,
        "ebitda_map": ebitda_map,
        "net_result_map": net_result_map,
        "bfr_map": bfr_map,
    }

results = run_full_model(settings, revenue_df, fixed_df, variable_df, payroll_df, capex_df, bfr_df)

# =========================================================
# EXPORT
# =========================================================
def build_excel_export(settings, results):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame([settings]).to_excel(writer, sheet_name="Parametres", index=False)
        revenue_df.to_excel(writer, sheet_name="Hyp_CA", index=False)
        fixed_df.to_excel(writer, sheet_name="Charges_fixes", index=False)
        variable_df.to_excel(writer, sheet_name="Charges_variables", index=False)
        payroll_df.to_excel(writer, sheet_name="Personnel", index=False)
        capex_df.to_excel(writer, sheet_name="CAPEX_Financement", index=False)
        bfr_df.to_excel(writer, sheet_name="Hyp_BFR", index=False)

        results["revenue_forecast"].to_excel(writer, sheet_name="CA_5_ans", index=False)
        results["fixed_forecast"].to_excel(writer, sheet_name="Fixes_5_ans", index=False)
        results["variable_forecast"].to_excel(writer, sheet_name="Variables_5_ans", index=False)
        results["payroll_forecast"]["detail"].to_excel(writer, sheet_name="Personnel_5_ans", index=False)
        results["capex_fin"]["schedule_df"].to_excel(writer, sheet_name="CAPEX_Schedule", index=False)
        results["bfr_forecast"].to_excel(writer, sheet_name="BFR_5_ans", index=False)
        results["pnl_df"].to_excel(writer, sheet_name="CR_5_ans", index=False)
        results["cashflow_df"].to_excel(writer, sheet_name="CashFlow_5_ans", index=False)

    output.seek(0)
    return output

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("EDDAQAQ EXP")
menu = st.sidebar.radio(
    "Navigation",
    [
        "Accueil",
        "Paramètres généraux",
        "Hypothèses CA",
        "Charges fixes",
        "Charges variables",
        "Personnel",
        "CAPEX & Financement",
        "BFR",
        "Compte de résultat 5 ans",
        "Cash flow 5 ans",
        "Analyse graphique",
        "Export Excel",
    ],
)

if st.sidebar.button("Déconnexion", use_container_width=True):
    st.session_state["authenticated"] = False
    st.rerun()

# =========================================================
# PAGE ACCUEIL
# =========================================================
if menu == "Accueil":
    st.title("Plateforme Prévisionnelle Clinique")
    st.write("Business plan, étude financière et cash flow prévisionnel sur 5 ans.")

    years = results["years"]
    pnl_df = results["pnl_df"]
    cash_df = results["cashflow_df"]

    y1 = years[0]
    y5 = years[-1]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("CA année 1", fmt_money(results["revenue_total_map"][y1]), settings.get("currency", "MAD"))
    with c2:
        metric_card("EBITDA année 1", fmt_money(results["ebitda_map"][y1]), settings.get("currency", "MAD"))
    with c3:
        metric_card("Résultat net année 1", fmt_money(results["net_result_map"][y1]), settings.get("currency", "MAD"))
    with c4:
        metric_card("BFR année 1", fmt_money(results["bfr_map"][y1]), settings.get("currency", "MAD"))

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        metric_card("CA année 5", fmt_money(results["revenue_total_map"][y5]), settings.get("currency", "MAD"))
    with c6:
        metric_card("Résultat net année 5", fmt_money(results["net_result_map"][y5]), settings.get("currency", "MAD"))
    with c7:
        metric_card("Trésorerie fin année 5", fmt_money(cash_df.iloc[-1]["Trésorerie fin"]), settings.get("currency", "MAD"))
    with c8:
        metric_card("CAPEX total", fmt_money(sum(results["capex_fin"]["capex_by_year"].values())), settings.get("currency", "MAD"))

    st.markdown("---")

    g1, g2 = st.columns(2)

    with g1:
        chart_df = pnl_df[["Année", "Chiffre d'affaires", "EBITDA", "Résultat net"]].melt(
            id_vars="Année",
            var_name="Indicateur",
            value_name="Montant"
        )
        fig = px.line(chart_df, x="Année", y="Montant", color="Indicateur", markers=True, title="CA, EBITDA et résultat net")
        st.plotly_chart(style_plot(fig), use_container_width=True)

    with g2:
        chart_df = cash_df[["Année", "Cash flow exploitation", "Cash flow investissement", "Cash flow financement", "Trésorerie fin"]].melt(
            id_vars="Année",
            var_name="Indicateur",
            value_name="Montant"
        )
        fig = px.bar(chart_df, x="Année", y="Montant", color="Indicateur", barmode="group", title="Cash flow et trésorerie")
        st.plotly_chart(style_plot(fig), use_container_width=True)

# =========================================================
# PARAMETRES
# =========================================================
elif menu == "Paramètres généraux":
    st.subheader("Paramètres généraux")

    with st.form("settings_form"):
        c1, c2 = st.columns(2)
        with c1:
            project_name = st.text_input("Nom du projet", value=settings.get("project_name", ""))
            company_name = st.text_input("Société / porteur", value=settings.get("company_name", ""))
            currency = st.text_input("Devise", value=settings.get("currency", "MAD"))
            start_year = st.number_input("Année de départ", value=int(settings.get("start_year", 2025)), step=1)
            nb_years = st.number_input("Nombre d'années", min_value=5, max_value=5, value=5, step=1)
            months_year1 = st.number_input("Nombre de mois d'activité année 1", min_value=1, max_value=12, value=int(settings.get("months_year1", 12)))

        with c2:
            opening_cash = st.number_input("Trésorerie d'ouverture", value=float(settings.get("opening_cash", 0.0)))
            inflation_pct = st.number_input("Inflation générale %", value=float(settings.get("inflation_pct", 2.0)))
            corp_tax_pct = st.number_input("IS %", value=float(settings.get("corp_tax_pct", 20.0)))
            solidarity_tax_pct = st.number_input("Taxe solidarité %", value=float(settings.get("solidarity_tax_pct", 1.5)))
            local_tax_y1 = st.number_input("Impôts et taxes d'exploitation Y1", value=float(settings.get("local_tax_y1", 0.0)))
            local_tax_growth_pct = st.number_input("Croissance impôts & taxes exploitation %", value=float(settings.get("local_tax_growth_pct", 2.0)))

        st.markdown("### Dividendes prévus")
        d1, d2, d3, d4, d5 = st.columns(5)
        dividend_y1 = d1.number_input("Y1", value=float(settings.get("dividend_y1", 0.0)), key="divy1")
        dividend_y2 = d2.number_input("Y2", value=float(settings.get("dividend_y2", 0.0)), key="divy2")
        dividend_y3 = d3.number_input("Y3", value=float(settings.get("dividend_y3", 0.0)), key="divy3")
        dividend_y4 = d4.number_input("Y4", value=float(settings.get("dividend_y4", 0.0)), key="divy4")
        dividend_y5 = d5.number_input("Y5", value=float(settings.get("dividend_y5", 0.0)), key="divy5")

        submitted = st.form_submit_button("Enregistrer", use_container_width=True)

    if submitted:
        new_settings = {
            "project_name": project_name,
            "company_name": company_name,
            "currency": currency,
            "start_year": int(start_year),
            "nb_years": int(nb_years),
            "months_year1": int(months_year1),
            "opening_cash": float(opening_cash),
            "inflation_pct": float(inflation_pct),
            "corp_tax_pct": float(corp_tax_pct),
            "solidarity_tax_pct": float(solidarity_tax_pct),
            "local_tax_y1": float(local_tax_y1),
            "local_tax_growth_pct": float(local_tax_growth_pct),
            "dividend_y1": float(dividend_y1),
            "dividend_y2": float(dividend_y2),
            "dividend_y3": float(dividend_y3),
            "dividend_y4": float(dividend_y4),
            "dividend_y5": float(dividend_y5),
        }
        save_section("settings", json.dumps(new_settings, ensure_ascii=False))
        st.success("Paramètres enregistrés.")
        st.rerun()

# =========================================================
# HYPOTHESES CA
# =========================================================
elif menu == "Hypothèses CA":
    st.subheader("Hypothèses de chiffre d'affaires")

    st.info(
        "Type = actes/jour : Base Y1 = nombre d'actes/jour | "
        "Type = lits*taux_occ : Base Y1 = taux d'occupation (ex 0.20) | "
        "Type = forfait_annuel : Base Y1 = montant annuel Y1"
    )

    edited = st.data_editor(
        revenue_df,
        use_container_width=True,
        num_rows="dynamic",
        key="revenue_editor"
    )

    if st.button("Enregistrer les hypothèses CA", use_container_width=True):
        save_section("revenue", json_dumps_df(edited))
        st.success("Hypothèses CA enregistrées.")
        st.rerun()

    st.markdown("### Aperçu calculé")
    st.dataframe(results["revenue_forecast"], use_container_width=True, hide_index=True)

# =========================================================
# CHARGES FIXES
# =========================================================
elif menu == "Charges fixes":
    st.subheader("Charges fixes")

    edited = st.data_editor(
        fixed_df,
        use_container_width=True,
        num_rows="dynamic",
        key="fixed_editor"
    )

    if st.button("Enregistrer les charges fixes", use_container_width=True):
        save_section("fixed_costs", json_dumps_df(edited))
        st.success("Charges fixes enregistrées.")
        st.rerun()

    st.markdown("### Aperçu calculé")
    st.dataframe(results["fixed_forecast"], use_container_width=True, hide_index=True)

# =========================================================
# CHARGES VARIABLES
# =========================================================
elif menu == "Charges variables":
    st.subheader("Charges variables")

    st.info("Mode = %CA pour une charge proportionnelle au chiffre d'affaires, ou Montant pour une charge annuelle directe.")

    edited = st.data_editor(
        variable_df,
        use_container_width=True,
        num_rows="dynamic",
        key="variable_editor"
    )

    if st.button("Enregistrer les charges variables", use_container_width=True):
        save_section("variable_costs", json_dumps_df(edited))
        st.success("Charges variables enregistrées.")
        st.rerun()

    st.markdown("### Aperçu calculé")
    st.dataframe(results["variable_forecast"], use_container_width=True, hide_index=True)

# =========================================================
# PERSONNEL
# =========================================================
elif menu == "Personnel":
    st.subheader("Salaires et charges sociales")

    edited = st.data_editor(
        payroll_df,
        use_container_width=True,
        num_rows="dynamic",
        key="payroll_editor"
    )

    if st.button("Enregistrer le personnel", use_container_width=True):
        save_section("payroll", json_dumps_df(edited))
        st.success("Personnel enregistré.")
        st.rerun()

    st.markdown("### Aperçu calculé")
    st.dataframe(results["payroll_forecast"]["detail"], use_container_width=True, hide_index=True)

# =========================================================
# CAPEX & FINANCEMENT
# =========================================================
elif menu == "CAPEX & Financement":
    st.subheader("Investissements, emprunts et leasing")

    st.info(
        "Année achat = 1 pour la première année du plan. "
        "Mode financement possible : Cash, Loan, Leasing. "
        "Taux financé % = part financée par emprunt ou leasing."
    )

    edited = st.data_editor(
        capex_df,
        use_container_width=True,
        num_rows="dynamic",
        key="capex_editor"
    )

    if st.button("Enregistrer CAPEX & financement", use_container_width=True):
        save_section("capex", json_dumps_df(edited))
        st.success("CAPEX & financement enregistrés.")
        st.rerun()

    st.markdown("### Schedule calculé")
    st.dataframe(results["capex_fin"]["schedule_df"], use_container_width=True, hide_index=True)

    years = results["years"]
    summary = pd.DataFrame({
        "Année": years,
        "CAPEX": [results["capex_fin"]["capex_by_year"][y] for y in years],
        "Amortissements": [results["capex_fin"]["depreciation_by_year"][y] for y in years],
        "Entrées emprunts": [results["capex_fin"]["loan_inflows_by_year"][y] for y in years],
        "Remboursement capital": [results["capex_fin"]["loan_principal_by_year"][y] for y in years],
        "Charges financières": [results["capex_fin"]["loan_interest_by_year"][y] for y in years],
        "Paiements leasing": [results["capex_fin"]["lease_payment_by_year"][y] for y in years],
        "Premier loyer / apport": [results["capex_fin"]["upfront_by_year"][y] for y in years],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# =========================================================
# BFR
# =========================================================
elif menu == "BFR":
    st.subheader("Hypothèses BFR")

    edited = st.data_editor(
        bfr_df,
        use_container_width=True,
        num_rows="dynamic",
        key="bfr_editor"
    )

    if st.button("Enregistrer les hypothèses BFR", use_container_width=True):
        save_section("bfr", json_dumps_df(edited))
        st.success("Hypothèses BFR enregistrées.")
        st.rerun()

    st.markdown("### BFR calculé")
    st.dataframe(results["bfr_forecast"], use_container_width=True, hide_index=True)

# =========================================================
# COMPTE DE RESULTAT
# =========================================================
elif menu == "Compte de résultat 5 ans":
    st.subheader("Compte de résultat prévisionnel sur 5 ans")
    st.dataframe(results["pnl_df"], use_container_width=True, hide_index=True)

# =========================================================
# CASH FLOW
# =========================================================
elif menu == "Cash flow 5 ans":
    st.subheader("Cash flow prévisionnel sur 5 ans")
    st.dataframe(results["cashflow_df"], use_container_width=True, hide_index=True)

# =========================================================
# ANALYSE
# =========================================================
elif menu == "Analyse graphique":
    st.subheader("Analyse graphique")

    pnl_df = results["pnl_df"]
    cash_df = results["cashflow_df"]
    bfr_df_calc = results["bfr_forecast"]
    years = results["years"]

    g1, g2 = st.columns(2)

    with g1:
        chart_df = pnl_df[["Année", "Chiffre d'affaires", "Charges variables", "Charges fixes", "Charges de personnel"]].melt(
            id_vars="Année",
            var_name="Indicateur",
            value_name="Montant"
        )
        fig = px.bar(chart_df, x="Année", y="Montant", color="Indicateur", barmode="group", title="Structure des charges et revenus")
        st.plotly_chart(style_plot(fig), use_container_width=True)

    with g2:
        fig = px.line(
            pnl_df,
            x="Année",
            y=["EBITDA", "Résultat net"],
            markers=True,
            title="Rentabilité"
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)

    g3, g4 = st.columns(2)

    with g3:
        fig = px.line(
            bfr_df_calc,
            x="Année",
            y=["BFR", "Variation BFR"],
            markers=True,
            title="BFR"
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)

    with g4:
        fig = px.line(
            cash_df,
            x="Année",
            y=["Cash flow exploitation", "Cash flow investissement", "Cash flow financement", "Trésorerie fin"],
            markers=True,
            title="Cash flow et trésorerie"
        )
        st.plotly_chart(style_plot(fig), use_container_width=True)

# =========================================================
# EXPORT
# =========================================================
elif menu == "Export Excel":
    st.subheader("Export du modèle")

    excel_file = build_excel_export(settings, results)

    st.download_button(
        "Télécharger le fichier Excel",
        data=excel_file,
        file_name="previsionnel_clinique_5_ans.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.success("L'export contient les hypothèses, le compte de résultat, le BFR et le cash flow.")