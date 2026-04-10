from io import BytesIO
from datetime import datetime
import math
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH_PNG = BASE_DIR / "assets" / "logo.png"
LOGO_PATH_JPG = BASE_DIR / "assets" / "logo.jpg"
LOGO_PATH_JPEG = BASE_DIR / "assets" / "logo.jpeg"

if LOGO_PATH_PNG.exists():
    LOGO_PATH = LOGO_PATH_PNG
elif LOGO_PATH_JPG.exists():
    LOGO_PATH = LOGO_PATH_JPG
elif LOGO_PATH_JPEG.exists():
    LOGO_PATH = LOGO_PATH_JPEG
else:
    LOGO_PATH = None

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
SAVE_FILE = DATA_DIR / "financial_inputs.json"

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Étude financière & Business Plan",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# MOT DE PASSE
# =========================================================
APP_PASSWORD = "EDDAQAQ2026"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("## Accès sécurisé")
    pwd = st.text_input("Mot de passe", type="password")

    if st.button("Se connecter"):
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect")

    st.stop()

YEAR_LABELS = [f"Année {i}" for i in range(1, 6)]
YEAR_NUMS = [1, 2, 3, 4, 5]
MONTH_LABELS = [
    "Jan", "Fév", "Mar", "Avr", "Mai", "Juin",
    "Juil", "Août", "Sep", "Oct", "Nov", "Déc"
]

# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], .main {
        background: #f7f4f2 !important;
        color: #2b2b2b !important;
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }

    .block-container {
        max-width: 1700px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e7deda !important;
    }

    [data-testid="stSidebar"] * {
        color: #2b2b2b !important;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #2b2b2b !important;
    }

    .card {
        background: #ffffff;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 16px;
        border: 1px solid #eadfda;
        box-shadow: 0 8px 24px rgba(120, 70, 70, 0.05);
    }

    .section-title {
        font-size: 1.08rem;
        font-weight: 800;
        color: #b7333a !important;
        margin-bottom: 0.65rem;
    }

    .sub-note {
        font-size: 0.92rem;
        color: #7a6f6a !important;
    }

    div[data-testid="stMetric"] {
        background: white !important;
        border: 1px solid #eadfda !important;
        border-radius: 16px !important;
        padding: 12px !important;
        box-shadow: 0 6px 18px rgba(120, 70, 70, 0.04);
    }

    .good-box, .warn-box, .risk-box {
        padding: 12px 14px;
        border-radius: 14px;
        margin-bottom: 10px;
        border: 1px solid transparent;
    }

    .good-box {
        background: #edf8f0;
        border-color: #a7d8b0;
        color: #27623a !important;
    }

    .warn-box {
        background: #fff6e8;
        border-color: #f0c97d;
        color: #8a5a00 !important;
    }

    .risk-box {
        background: #fff0f0;
        border-color: #e7aaaa;
        color: #8b2222 !important;
    }

    .stButton > button,
    .stDownloadButton > button,
    button[kind="primary"] {
        background: #b7333a !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    button[kind="primary"]:hover {
        background: #9e2c33 !important;
        color: white !important;
    }

    button[data-baseweb="tab"] {
        color: #7a6f6a !important;
        font-weight: 700 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #b7333a !important;
        border-bottom: 2px solid #b7333a !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #eadfda;
        border-radius: 14px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================
def fmt_mad(x) -> str:
    try:
        return f"{float(x):,.0f} MAD".replace(",", " ")
    except Exception:
        return "0 MAD"


def fmt_pct(x) -> str:
    try:
        return f"{float(x):.1%}"
    except Exception:
        return "0.0%"


def n(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def i(x, default=0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def safe_div(a, b) -> float:
    if b in [0, None] or pd.isna(b):
        return 0.0
    return float(a) / float(b)


def growth_series(year1_value: float, growths: list[float]) -> list[float]:
    vals = [year1_value]
    current = year1_value
    for g in growths:
        current = current * (1 + g)
        vals.append(current)
    return vals


def normalize_percent_list(values):
    arr = np.array(values, dtype=float)
    total = arr.sum()
    if total <= 0:
        return np.array([1 / 12] * 12, dtype=float)
    return arr / total


def to_year_columns_df(row_dict: dict, first_col_name: str = "Rubrique") -> pd.DataFrame:
    rows = []
    for label, values in row_dict.items():
        row = {first_col_name: label}
        for idx, y in enumerate(YEAR_LABELS):
            row[y] = values[idx] if idx < len(values) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def money_style(df: pd.DataFrame, non_money_cols=None):
    non_money_cols = non_money_cols or []
    formats = {}
    for col in df.columns:
        if col not in non_money_cols:
            formats[col] = "{:,.0f}"
    return df.style.format(formats)


def annuity_payment(principal: float, annual_rate: float, months: int) -> float:
    if principal <= 0 or months <= 0:
        return 0.0
    m_rate = annual_rate / 12
    if abs(m_rate) < 1e-12:
        return principal / months
    return principal * (m_rate / (1 - (1 + m_rate) ** (-months)))


def build_loan_schedule(
    principal: float,
    annual_rate: float,
    months: int,
    deferment_months: int = 0,
    projection_years: int = 5,
    name: str = "Prêt"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows = []
    annual_rows = []

    if principal <= 0 or months <= 0:
        for y in YEAR_NUMS:
            annual_rows.append({
                "Source": name,
                "Année": y,
                "Mensualité": 0.0,
                "Annuité": 0.0,
                "Intérêts": 0.0,
                "Remboursement capital": 0.0,
                "Capital restant dû": 0.0,
            })
        return pd.DataFrame(annual_rows), pd.DataFrame(monthly_rows)

    balance = principal
    pay = annuity_payment(principal, annual_rate, months)
    m_rate = annual_rate / 12

    total_projection_months = projection_years * 12
    for m in range(1, total_projection_months + 1):
        if m <= deferment_months and balance > 0:
            interest = balance * m_rate
            principal_paid = 0.0
            mensualite = interest
        elif balance > 1e-8:
            interest = balance * m_rate
            principal_paid = min(pay - interest, balance)
            mensualite = principal_paid + interest
            balance = max(balance - principal_paid, 0.0)
        else:
            interest = 0.0
            principal_paid = 0.0
            mensualite = 0.0

        monthly_rows.append({
            "Source": name,
            "Mois_index": m,
            "Année": math.ceil(m / 12),
            "Mensualité": mensualite,
            "Intérêts": interest,
            "Remboursement capital": principal_paid,
            "Capital restant dû": balance,
        })

    monthly_df = pd.DataFrame(monthly_rows)

    for y in YEAR_NUMS:
        tmp = monthly_df[monthly_df["Année"] == y]
        annual_rows.append({
            "Source": name,
            "Année": y,
            "Mensualité": tmp["Mensualité"].iloc[0] if not tmp.empty else 0.0,
            "Annuité": tmp["Mensualité"].sum(),
            "Intérêts": tmp["Intérêts"].sum(),
            "Remboursement capital": tmp["Remboursement capital"].sum(),
            "Capital restant dû": tmp["Capital restant dû"].iloc[-1] if not tmp.empty else 0.0,
        })

    return pd.DataFrame(annual_rows), monthly_df


def linear_amortization(amount: float, duration_years: int, projection_years: int = 5) -> list[float]:
    if amount <= 0 or duration_years <= 0:
        return [0.0] * projection_years
    annual = amount / duration_years
    return [annual if y <= duration_years else 0.0 for y in range(1, projection_years + 1)]


def solidarity_rate(ebt):
    if ebt <= 1_000_000:
        return 0.0
    if ebt <= 5_000_000:
        return 0.015
    if ebt <= 10_000_000:
        return 0.025
    if ebt <= 40_000_000:
        return 0.035
    return 0.05


def build_default_investments():
    return pd.DataFrame([
        {"Rubrique": "Frais d'établissement", "Montant": 80000.0, "Durée amort. (ans)": 5},
        {"Rubrique": "Frais d'étude", "Montant": 100000.0, "Durée amort. (ans)": 5},
        {"Rubrique": "Logiciels / formations", "Montant": 100000.0, "Durée amort. (ans)": 3},
        {"Rubrique": "Dépôt marque / brevet / modèle", "Montant": 20000.0, "Durée amort. (ans)": 5},
        {"Rubrique": "Droits d'entrée", "Montant": 0.0, "Durée amort. (ans)": 5},
        {"Rubrique": "Achat fonds de commerce / parts", "Montant": 0.0, "Durée amort. (ans)": 10},
        {"Rubrique": "Droit au bail", "Montant": 500000.0, "Durée amort. (ans)": 10},
        {"Rubrique": "Caution / dépôt de garantie", "Montant": 300000.0, "Durée amort. (ans)": 5},
        {"Rubrique": "Frais de dossier", "Montant": 30000.0, "Durée amort. (ans)": 5},
        {"Rubrique": "Frais de notaire / avocat", "Montant": 70000.0, "Durée amort. (ans)": 5},
        {"Rubrique": "Enseigne / communication de lancement", "Montant": 120000.0, "Durée amort. (ans)": 4},
        {"Rubrique": "Achat immobilier", "Montant": 1000000.0, "Durée amort. (ans)": 20},
        {"Rubrique": "Construction / aménagement", "Montant": 4500000.0, "Durée amort. (ans)": 10},
        {"Rubrique": "Terrassement", "Montant": 300000.0, "Durée amort. (ans)": 10},
        {"Rubrique": "Matériel / équipements", "Montant": 12000000.0, "Durée amort. (ans)": 7},
        {"Rubrique": "Stock initial", "Montant": 800000.0, "Durée amort. (ans)": 1},
        {"Rubrique": "Trésorerie de départ", "Montant": 4000000.0, "Durée amort. (ans)": 0},
    ])


def build_default_financing():
    return pd.DataFrame([
        {"Source": "Apport personnel / familial", "Type": "Fonds propres", "Montant": 1010000.0, "Taux annuel": 0.0, "Durée (mois)": 0, "Différé (mois)": 0},
        {"Source": "Apports en nature", "Type": "Fonds propres", "Montant": 0.0, "Taux annuel": 0.0, "Durée (mois)": 0, "Différé (mois)": 0},
        {"Source": "Compte courant associé", "Type": "Quasi-fonds propres", "Montant": 0.0, "Taux annuel": 0.0, "Durée (mois)": 0, "Différé (mois)": 0},
        {"Source": "Prêt n°1", "Type": "Emprunt", "Montant": 0.0, "Taux annuel": 0.06, "Durée (mois)": 120, "Différé (mois)": 0},
        {"Source": "Prêt n°2", "Type": "Emprunt", "Montant": 0.0, "Taux annuel": 0.05, "Durée (mois)": 84, "Différé (mois)": 0},
        {"Source": "Prêt n°3", "Type": "Emprunt", "Montant": 22910000.0, "Taux annuel": 0.05, "Durée (mois)": 84, "Différé (mois)": 0},
        {"Source": "Subvention", "Type": "Subvention", "Montant": 0.0, "Taux annuel": 0.0, "Durée (mois)": 0, "Différé (mois)": 0},
        {"Source": "Autre financement", "Type": "Autre", "Montant": 0.0, "Taux annuel": 0.0, "Durée (mois)": 0, "Différé (mois)": 0},
    ])


def build_default_leasing():
    return pd.DataFrame([
        {"Contrat": "LRM1", "Montant financé TTC": 120000.0, "Redevance mensuelle TTC": 2592.0, "Nombre de mois": 60, "Mois de départ": 1},
        {"Contrat": "LRM2", "Montant financé TTC": 90000.0, "Redevance mensuelle TTC": 1944.0, "Nombre de mois": 60, "Mois de départ": 1},
        {"Contrat": "LRM3", "Montant financé TTC": 75000.0, "Redevance mensuelle TTC": 1620.0, "Nombre de mois": 60, "Mois de départ": 1},
        {"Contrat": "LRM4", "Montant financé TTC": 60000.0, "Redevance mensuelle TTC": 1296.0, "Nombre de mois": 60, "Mois de départ": 1},
        {"Contrat": "FP1", "Montant financé TTC": 30000.0, "Redevance mensuelle TTC": 648.0, "Nombre de mois": 48, "Mois de départ": 1},
        {"Contrat": "FP2", "Montant financé TTC": 250000.0, "Redevance mensuelle TTC": 5400.0, "Nombre de mois": 48, "Mois de départ": 1},
        {"Contrat": "FP3", "Montant financé TTC": 20000.0, "Redevance mensuelle TTC": 432.0, "Nombre de mois": 48, "Mois de départ": 1},
        {"Contrat": "FP4", "Montant financé TTC": 15000.0, "Redevance mensuelle TTC": 324.0, "Nombre de mois": 48, "Mois de départ": 1},
        {"Contrat": "FP5", "Montant financé TTC": 10000.0, "Redevance mensuelle TTC": 216.0, "Nombre de mois": 48, "Mois de départ": 1},
    ])


def build_default_fixed_charges():
    return pd.DataFrame([
        {"Rubrique": "Assurances", "Année 1": 40000.0, "Année 2": 45000.0, "Année 3": 45000.0, "Année 4": 45000.0, "Année 5": 45000.0},
        {"Rubrique": "Téléphone / internet", "Année 1": 15000.0, "Année 2": 15000.0, "Année 3": 15000.0, "Année 4": 20000.0, "Année 5": 20000.0},
        {"Rubrique": "Autres abonnements", "Année 1": 10000.0, "Année 2": 10000.0, "Année 3": 10000.0, "Année 4": 10000.0, "Année 5": 10000.0},
        {"Rubrique": "Carburant / transports", "Année 1": 25000.0, "Année 2": 25000.0, "Année 3": 30000.0, "Année 4": 35000.0, "Année 5": 40000.0},
        {"Rubrique": "Déplacements / hébergement", "Année 1": 15000.0, "Année 2": 15000.0, "Année 3": 15000.0, "Année 4": 15000.0, "Année 5": 15000.0},
        {"Rubrique": "Eau / électricité / gaz", "Année 1": 100000.0, "Année 2": 100000.0, "Année 3": 100000.0, "Année 4": 100000.0, "Année 5": 100000.0},
        {"Rubrique": "Mutuelle", "Année 1": 10000.0, "Année 2": 10000.0, "Année 3": 10000.0, "Année 4": 10000.0, "Année 5": 10000.0},
        {"Rubrique": "Fournitures diverses", "Année 1": 90322.92, "Année 2": 8840050.0, "Année 3": 12290050.0, "Année 4": 15493375.0, "Année 5": 16075562.5},
        {"Rubrique": "Entretien matériel / vêtements", "Année 1": 40000.0, "Année 2": 40000.0, "Année 3": 40000.0, "Année 4": 40000.0, "Année 5": 40000.0},
        {"Rubrique": "Nettoyage locaux", "Année 1": 30000.0, "Année 2": 30000.0, "Année 3": 30000.0, "Année 4": 30000.0, "Année 5": 30000.0},
        {"Rubrique": "Publicité / communication", "Année 1": 50000.0, "Année 2": 50000.0, "Année 3": 50000.0, "Année 4": 50000.0, "Année 5": 50000.0},
        {"Rubrique": "Loyer et charges locatives", "Année 1": 250000.0, "Année 2": 250000.0, "Année 3": 250000.0, "Année 4": 250000.0, "Année 5": 250000.0},
        {"Rubrique": "Frais bancaires", "Année 1": 10000.0, "Année 2": 10000.0, "Année 3": 10000.0, "Année 4": 10000.0, "Année 5": 10000.0},
        {"Rubrique": "Taxes diverses", "Année 1": 20000.0, "Année 2": 20000.0, "Année 3": 20000.0, "Année 4": 20000.0, "Année 5": 20000.0},
        {"Rubrique": "Expert-comptable", "Année 1": 10000.0, "Année 2": 10000.0, "Année 3": 10000.0, "Année 4": 10000.0, "Année 5": 10000.0},
    ])


def build_default_salary_table():
    return pd.DataFrame([
        {"Poste": "Directeurs et managers", "Salaire brut mensuel": 105400.0, "Année 1": 2, "Année 2": 4, "Année 3": 4, "Année 4": 4, "Année 5": 4},
        {"Poste": "Directeur médical", "Salaire brut mensuel": 34000.0, "Année 1": 1, "Année 2": 1, "Année 3": 1, "Année 4": 1, "Année 5": 1},
        {"Poste": "Nutritionniste", "Salaire brut mensuel": 25500.0, "Année 1": 0, "Année 2": 1, "Année 3": 1, "Année 4": 1, "Année 5": 1},
        {"Poste": "Directeur de soins", "Salaire brut mensuel": 25500.0, "Année 1": 1, "Année 2": 1, "Année 3": 1, "Année 4": 1, "Année 5": 1},
        {"Poste": "Personnel paramédical", "Salaire brut mensuel": 133650.0 / 38.0, "Année 1": 38, "Année 2": 70, "Année 3": 81, "Année 4": 92, "Année 5": 104},
        {"Poste": "Personnel administratif et support", "Salaire brut mensuel": 197775.0 / 27.0, "Année 1": 27, "Année 2": 43, "Année 3": 43, "Année 4": 43, "Année 5": 43},
    ])


def build_default_monthly_distribution():
    return pd.DataFrame({
        "Mois": MONTH_LABELS,
        "% CA marchandises": [100 / 12] * 12,
        "% CA services": [100 / 12] * 12,
    })


def build_default_bfr_days():
    return pd.DataFrame([
        {"Rubrique": "Stock (jours d'achats)", "Année 1": 15, "Année 2": 15, "Année 3": 15, "Année 4": 15, "Année 5": 15},
        {"Rubrique": "Créances AMO (jours)", "Année 1": 120, "Année 2": 120, "Année 3": 120, "Année 4": 120, "Année 5": 120},
        {"Rubrique": "Créances privées (jours)", "Année 1": 120, "Année 2": 120, "Année 3": 120, "Année 4": 120, "Année 5": 120},
        {"Rubrique": "Créances sans couverture (jours)", "Année 1": 0, "Année 2": 0, "Année 3": 0, "Année 4": 0, "Année 5": 0},
        {"Rubrique": "Autres créances (jours)", "Année 1": 30, "Année 2": 30, "Année 3": 30, "Année 4": 30, "Année 5": 30},
        {"Rubrique": "Dettes fournisseurs (jours)", "Année 1": 60, "Année 2": 60, "Année 3": 60, "Année 4": 60, "Année 5": 60},
        {"Rubrique": "Autres dettes (jours)", "Année 1": 30, "Année 2": 30, "Année 3": 30, "Année 4": 30, "Année 5": 30},
    ])


def build_default_taxes_table():
    return pd.DataFrame([
        {"Rubrique": "IS", "Année 1": 0.20, "Année 2": 0.20, "Année 3": 0.20, "Année 4": 0.20, "Année 5": 0.20},
        {"Rubrique": "Taxe services communaux", "Année 1": 0.105, "Année 2": 0.105, "Année 3": 0.105, "Année 4": 0.105, "Année 5": 0.105},
        {"Rubrique": "Taxe professionnelle", "Année 1": 0.20, "Année 2": 0.20, "Année 3": 0.20, "Année 4": 0.20, "Année 5": 0.20},
        {"Rubrique": "Droits d'enregistrement / timbre", "Année 1": 0.0025, "Année 2": 0.0025, "Année 3": 0.0025, "Année 4": 0.0025, "Année 5": 0.0025},
        {"Rubrique": "Taxe enseigne", "Année 1": 200000.0, "Année 2": 200000.0, "Année 3": 200000.0, "Année 4": 200000.0, "Année 5": 200000.0},
        {"Rubrique": "Charges non courantes (% du CA)", "Année 1": 0.001, "Année 2": 0.001, "Année 3": 0.001, "Année 4": 0.001, "Année 5": 0.001},
    ])


def build_default_ca_services_table():
    return pd.DataFrame([
        {
            "Désignation": "Consultation générale",
            "Nombre de jours ouvrés par an": 312,
            "Tarif par consultation": 250.0,
            "Part clinique (%)": 55.0,
            "Part médecin (%)": 45.0,
        },
        {
            "Désignation": "Consultation spécialisée",
            "Nombre de jours ouvrés par an": 312,
            "Tarif par consultation": 400.0,
            "Part clinique (%)": 55.0,
            "Part médecin (%)": 45.0,
        },
    ])


def df_to_records(df: pd.DataFrame):
    tmp = df.copy()
    return tmp.replace({np.nan: None}).to_dict(orient="records")


def records_to_df(records, default_df: pd.DataFrame):
    try:
        df = pd.DataFrame(records)
        if df.empty:
            return default_df.copy()
        for col in default_df.columns:
            if col not in df.columns:
                df[col] = default_df[col].iloc[0] if len(default_df) > 0 else None
        return df[default_df.columns].copy()
    except Exception:
        return default_df.copy()


def save_all_inputs():
    payload = {
        "investments_df": df_to_records(st.session_state["investments_df"]),
        "financing_df": df_to_records(st.session_state["financing_df"]),
        "leasing_df": df_to_records(st.session_state["leasing_df"]),
        "fixed_df": df_to_records(st.session_state["fixed_df"]),
        "salary_df": df_to_records(st.session_state["salary_df"]),
        "ca_services_df": df_to_records(st.session_state["ca_services_df"]),
        "monthly_dist_df": df_to_records(st.session_state["monthly_dist_df"]),
        "bfr_days_df": df_to_records(st.session_state["bfr_days_df"]),
        "taxes_df": df_to_records(st.session_state["taxes_df"]),
    }
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_all_inputs():
    if not SAVE_FILE.exists():
        return None
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def reset_all_inputs():
    st.session_state["investments_df"] = build_default_investments()
    st.session_state["financing_df"] = build_default_financing()
    st.session_state["leasing_df"] = build_default_leasing()
    st.session_state["fixed_df"] = build_default_fixed_charges()
    st.session_state["salary_df"] = build_default_salary_table()
    st.session_state["ca_services_df"] = pd.DataFrame(columns=[
        "Désignation",
        "Nombre de jours ouvrés par an",
        "Tarif par consultation",
        "Part clinique (%)",
        "Part médecin (%)",
    ])
    st.session_state["monthly_dist_df"] = build_default_monthly_distribution()
    st.session_state["bfr_days_df"] = build_default_bfr_days()
    st.session_state["taxes_df"] = build_default_taxes_table()


def ensure_state():
    defaults = {
        "investments_df": build_default_investments(),
        "financing_df": build_default_financing(),
        "leasing_df": build_default_leasing(),
        "fixed_df": build_default_fixed_charges(),
        "salary_df": build_default_salary_table(),
        "ca_services_df": build_default_ca_services_table(),
        "monthly_dist_df": build_default_monthly_distribution(),
        "bfr_days_df": build_default_bfr_days(),
        "taxes_df": build_default_taxes_table(),
    }

    loaded = load_all_inputs()

    for key, value in defaults.items():
        if key not in st.session_state:
            if loaded and key in loaded:
                st.session_state[key] = records_to_df(loaded[key], value)
            else:
                st.session_state[key] = value


def render_editor_stateful(state_key, editor_key, column_config=None, num_rows="dynamic"):
    edited_df = st.data_editor(
        st.session_state[state_key].copy(),
        use_container_width=True,
        num_rows=num_rows,
        key=editor_key,
        column_config=column_config or {},
    )
    st.session_state[state_key] = edited_df.copy()
    return edited_df


ensure_state()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    if LOGO_PATH and LOGO_PATH.exists():
        st.markdown(
            """
            <div style="display:flex; justify-content:flex-start; align-items:center; padding:6px 0 14px 0;">
            """,
            unsafe_allow_html=True
        )
        st.image(str(LOGO_PATH), width=220)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Logo introuvable dans assets/logo.png")

    st.markdown("## Paramètres généraux")
    project_name = st.text_input("Nom du projet", value="Clinique Multidisciplinaire")
    project_holder = st.text_input("Porteur du projet", value="Société CLINIQUE CAMILIA")
    legal_status = st.selectbox(
        "Statut juridique",
        ["Micro-entreprise", "Entreprise individuelle au réel (IR)", "EURL (IS)", "SARL (IS)", "SAS (IS)", "SASU (IS)", "Autre"]
    )
    activity_nature = st.selectbox("Nature de l'activité", ["Services", "Marchandises", "Mixte"])
    city = st.text_input("Ville / commune", value="ERRAHMA")
    phone = st.text_input("Téléphone", value="")
    email = st.text_input("E-mail", value="")
    months_activity_y1 = st.number_input("Mois d'activité année 1", min_value=1, max_value=12, value=8)
    employer_social_rate = st.number_input("Taux charges sociales employeur (%)", min_value=0.0, value=21.09, step=0.1) / 100
    amo_share = st.number_input("Part CA couverte AMO (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0) / 100
    private_share = st.number_input("Part CA couverte assurances privées (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0) / 100
    uninsured_share = st.number_input("Part CA sans couverture (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0) / 100
    cash_share = st.number_input("Part encaissée en espèces (%)", min_value=0.0, max_value=100.0, value=5.0, step=1.0) / 100
    cheque_share = st.number_input("Part encaissée par chèque (%)", min_value=0.0, max_value=100.0, value=3.0, step=1.0) / 100
    card_share = st.number_input("Part encaissée par carte (%)", min_value=0.0, max_value=100.0, value=2.0, step=1.0) / 100
    auto_exempt_prof_tax = st.checkbox("Exonération taxe professionnelle sur 5 ans", value=True)
    st.caption("Les tableaux du plan financier se remplissent automatiquement à partir des données saisies ci-dessous.")

# =========================================================
# HERO
# =========================================================
st.markdown(
    f"""
    <div style="
        background: linear-gradient(rgba(255,255,255,0.82), rgba(255,255,255,0.82)),
                    url('https://images.unsplash.com/photo-1554224155-6726b3ff858f?q=80&w=1600&auto=format&fit=crop') center/cover no-repeat;
        border-radius: 24px;
        padding: 44px 32px;
        border: 1px solid #eadfda;
        margin-bottom: 18px;
        text-align:center;
    ">
        <div style="font-size:16px; font-weight:700; color:#b7333a; margin-bottom:10px;">
            EDDAQAQ EXPERTISES
        </div>
        <div style="font-size:52px; font-weight:800; line-height:1.05; color:#b7333a; margin-bottom:12px;">
            Expert consultant<br>Pour votre Entreprise
        </div>
        <div style="font-size:16px; color:#645a56;">
            Étude financière prévisionnelle sur 5 ans - {project_name}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TABS
# =========================================================
tab_input, tab_print, tab_report, tab_calc, tab_export = st.tabs([
    "1. Données à saisir",
    "2. Plan financier à imprimer",
    "3. Contrôle global & analyse",
    "4. Base de calculs / KPI",
    "5. Exports Excel / PDF",
])

# =========================================================
# INPUT TAB
# =========================================================
with tab_input:
    action_col1, action_col2, _ = st.columns([1.2, 1.2, 4])

    with action_col1:
        if st.button("💾 Enregistrer les données", use_container_width=True):
            save_all_inputs()
            st.success("Les données ont été enregistrées avec succès.")

    with action_col2:
        if st.button("🧹 Vider / Réinitialiser", use_container_width=True):
            reset_all_inputs()
            if SAVE_FILE.exists():
                SAVE_FILE.unlink()
            st.warning("Toutes les colonnes ont été réinitialisées.")
            st.rerun()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">A. Identité du projet</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.text_input("Nom / raison sociale", value=project_holder, disabled=True, key="id_holder")
    with c2:
        st.text_input("Projet / activité", value=project_name, disabled=True, key="id_project")
    with c3:
        st.text_input("Ville", value=city, disabled=True, key="id_city")
    with c4:
        st.text_input("Statut juridique", value=legal_status, disabled=True, key="id_legal")
    st.markdown("<p class='sub-note'>Cette zone est reprise automatiquement dans les exports et le rapport.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">B. Besoins de démarrage</div>', unsafe_allow_html=True)
    st.markdown("<p class='sub-note'>Renseigne ici uniquement les montants et les durées. L’amortissement, la synthèse investissements et le plan de financement se calculent automatiquement.</p>", unsafe_allow_html=True)
    investments_df = render_editor_stateful("investments_df", "editor_investments", num_rows="dynamic")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">C. Financement du projet</div>', unsafe_allow_html=True)
    financing_df = render_editor_stateful(
        "financing_df",
        "editor_financing",
        num_rows="dynamic",
        column_config={
            "Type": st.column_config.SelectboxColumn(
                "Type",
                options=["Fonds propres", "Quasi-fonds propres", "Emprunt", "Subvention", "Autre"]
            )
        }
    )
    st.markdown("<p class='sub-note'>Pour les lignes de type Emprunt, renseigne le montant, le taux, la durée en mois et le différé éventuel.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">D. Contrats de crédit-bail / leasing</div>', unsafe_allow_html=True)
    leasing_df = render_editor_stateful("leasing_df", "editor_leasing", num_rows="dynamic")
    st.markdown("<p class='sub-note'>L’outil calcule automatiquement les redevances annuelles par année à partir des mensualités TTC.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">E. Salaires et charges sociales</div>', unsafe_allow_html=True)
    salary_df = render_editor_stateful("salary_df", "editor_salary", num_rows="dynamic")
    st.markdown("<p class='sub-note'>Saisir le salaire brut mensuel et l’effectif par année. La masse salariale et les charges sociales se calculent seules.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">F. Charges fixes hors salaires</div>', unsafe_allow_html=True)
    fixed_df = render_editor_stateful("fixed_df", "editor_fixed", num_rows="dynamic")
    st.markdown("<p class='sub-note'>Tu peux modifier directement chaque année. Donc pas d’inversion lignes/colonnes dans le résultat final.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">G. Chiffre d’affaires prévisionnel</div>', unsafe_allow_html=True)
    st.markdown("<p class='sub-note'>Saisis ici les consultations ou actes. La part clinique et la part médecin se calculent automatiquement sur la base du tarif et des pourcentages.</p>", unsafe_allow_html=True)

    ca_services_df = render_editor_stateful(
        "ca_services_df",
        "editor_ca_services",
        num_rows="dynamic",
        column_config={
            "Désignation": st.column_config.TextColumn("Désignations"),
            "Nombre de jours ouvrés par an": st.column_config.NumberColumn("Nombre de jours ouvrés par an", min_value=0, step=1),
            "Tarif par consultation": st.column_config.NumberColumn("Tarif par consultation", min_value=0.0, step=10.0, format="%.2f"),
            "Part clinique (%)": st.column_config.NumberColumn("Part clinique", min_value=0.0, max_value=100.0, step=1.0, format="%.1f"),
            "Part médecin (%)": st.column_config.NumberColumn("Part médecin", min_value=0.0, max_value=100.0, step=1.0, format="%.1f"),
        }
    )

    ca_services_preview = ca_services_df.copy()
    for col in ["Nombre de jours ouvrés par an", "Tarif par consultation", "Part clinique (%)", "Part médecin (%)"]:
        ca_services_preview[col] = pd.to_numeric(ca_services_preview[col], errors="coerce").fillna(0.0)

    if not ca_services_preview.empty:
        ca_services_preview["CA brut annuel"] = (
            ca_services_preview["Nombre de jours ouvrés par an"] *
            ca_services_preview["Tarif par consultation"]
        )
        ca_services_preview["CA part clinique"] = (
            ca_services_preview["CA brut annuel"] *
            ca_services_preview["Part clinique (%)"] / 100
        )
        ca_services_preview["CA part médecin"] = (
            ca_services_preview["CA brut annuel"] *
            ca_services_preview["Part médecin (%)"] / 100
        )
        ca_services_preview["Total répartition %"] = (
            ca_services_preview["Part clinique (%)"] +
            ca_services_preview["Part médecin (%)"]
        )

    st.markdown("<p class='sub-note'>Aperçu des montants calculés :</p>", unsafe_allow_html=True)
    st.dataframe(ca_services_preview, use_container_width=True)

    invalid_rows = ca_services_preview[ca_services_preview["Total répartition %"] != 100] if not ca_services_preview.empty else pd.DataFrame()
    if not invalid_rows.empty:
        st.warning("Attention : pour certaines lignes, Part clinique + Part médecin est différente de 100%.")

    dist_df = render_editor_stateful("monthly_dist_df", "editor_month_dist", num_rows="fixed")
    st.markdown("</div>", unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">H. Charges variables & BFR</div>', unsafe_allow_html=True)

        goods_purchase_rate = st.number_input("Coût d’achat marchandises / CA marchandises (%)", min_value=0.0, value=57.0, step=1.0) / 100
        medical_consumables_rate = st.number_input("Consommables médicaux / CA services (%)", min_value=0.0, value=20.0, step=1.0) / 100
        office_supplies_rate = st.number_input("Fournitures bureau / CA total (%)", min_value=0.0, value=2.0, step=0.5) / 100
        other_variable_rate = st.number_input("Autres variables / CA total (%)", min_value=0.0, value=0.0, step=0.5) / 100

        bfr_days_df = render_editor_stateful("bfr_days_df", "editor_bfr_days", num_rows="dynamic")
        st.markdown("</div>", unsafe_allow_html=True)

    with cc2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">I. Impôts, taxes et paramètres</div>', unsafe_allow_html=True)
        taxes_df = render_editor_stateful("taxes_df", "editor_taxes", num_rows="dynamic")

        waste_kg_month = st.number_input("Déchets médicaux (kg / mois)", min_value=0.0, value=3500.0, step=100.0)
        waste_cost_per_kg = st.number_input("Coût traitement déchets par kg", min_value=0.0, value=6.0, step=0.5)
        locative_base_y1 = st.number_input("Base valeur locative fiscale année 1", min_value=0.0, value=967600.0, step=1000.0)

        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# CALCULS
# =========================================================
investments_df = st.session_state["investments_df"].copy()
financing_df = st.session_state["financing_df"].copy()
leasing_df = st.session_state["leasing_df"].copy()
salary_df = st.session_state["salary_df"].copy()
fixed_df = st.session_state["fixed_df"].copy()
ca_services_df = st.session_state["ca_services_df"].copy()
dist_df = st.session_state["monthly_dist_df"].copy()
bfr_days_df = st.session_state["bfr_days_df"].copy()
taxes_df = st.session_state["taxes_df"].copy()

for df in [investments_df, financing_df, leasing_df, salary_df, fixed_df, dist_df, bfr_days_df, taxes_df]:
    for col in df.columns:
        if col not in ["Rubrique", "Source", "Type", "Contrat", "Poste", "Mois"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

for col in ["Nombre de jours ouvrés par an", "Tarif par consultation", "Part clinique (%)", "Part médecin (%)"]:
    if col in ca_services_df.columns:
        ca_services_df[col] = pd.to_numeric(ca_services_df[col], errors="coerce").fillna(0.0)

investment_rows = []
amort_rows = []
amort_totals = [0.0] * 5

cash_start_need = 0.0
capex_total_excl_cash = 0.0
total_startup_needs = 0.0

for _, r in investments_df.iterrows():
    label = str(r["Rubrique"])
    amount = n(r["Montant"])
    duration = i(r["Durée amort. (ans)"], 0)

    total_startup_needs += amount

    if label.strip().lower() == "trésorerie de départ":
        cash_start_need += amount
    else:
        capex_total_excl_cash += amount

    investment_rows.append({"Rubrique": label, "Montant": amount})

    amort_values = linear_amortization(amount, duration, 5) if duration > 0 else [0.0] * 5
    for idx in range(5):
        amort_totals[idx] += amort_values[idx]

    amort_row = {"Rubrique": label, "Montant": amount, "Durée amort. (ans)": duration}
    for idx, y in enumerate(YEAR_LABELS):
        amort_row[y] = amort_values[idx]
    amort_rows.append(amort_row)

df_invest = pd.DataFrame(investment_rows)
df_amort = pd.DataFrame(amort_rows)

total_resources = financing_df["Montant"].sum()

loan_annual_list = []
loan_monthly_list = []
for _, r in financing_df.iterrows():
    if str(r["Type"]) == "Emprunt" and n(r["Montant"]) > 0:
        annual_df, monthly_df = build_loan_schedule(
            principal=n(r["Montant"]),
            annual_rate=n(r["Taux annuel"]),
            months=i(r["Durée (mois)"]),
            deferment_months=i(r["Différé (mois)"]),
            projection_years=5,
            name=str(r["Source"])
        )
        loan_annual_list.append(annual_df)
        loan_monthly_list.append(monthly_df)

if loan_annual_list:
    loans_annual_detail = pd.concat(loan_annual_list, ignore_index=True)
    loans_monthly_detail = pd.concat(loan_monthly_list, ignore_index=True)
else:
    loans_annual_detail = pd.DataFrame(columns=["Source", "Année", "Mensualité", "Annuité", "Intérêts", "Remboursement capital", "Capital restant dû"])
    loans_monthly_detail = pd.DataFrame(columns=["Source", "Mois_index", "Année", "Mensualité", "Intérêts", "Remboursement capital", "Capital restant dû"])

loan_summary = []
for y in YEAR_NUMS:
    tmp = loans_annual_detail[loans_annual_detail["Année"] == y] if not loans_annual_detail.empty else pd.DataFrame()
    loan_summary.append({
        "Année": y,
        "Mensualité": tmp["Mensualité"].sum() if not tmp.empty else 0.0,
        "Annuité": tmp["Annuité"].sum() if not tmp.empty else 0.0,
        "Intérêts": tmp["Intérêts"].sum() if not tmp.empty else 0.0,
        "Remboursement capital": tmp["Remboursement capital"].sum() if not tmp.empty else 0.0,
        "Capital restant dû": tmp["Capital restant dû"].sum() if not tmp.empty else 0.0,
    })
df_loans = pd.DataFrame(loan_summary)

leasing_annual = {y: 0.0 for y in YEAR_NUMS}
for _, r in leasing_df.iterrows():
    start_month = max(1, i(r["Mois de départ"], 1))
    nb_months = i(r["Nombre de mois"])
    monthly_ttc = n(r["Redevance mensuelle TTC"])
    for m in range(start_month, start_month + nb_months):
        year = math.ceil(m / 12)
        if year in leasing_annual:
            leasing_annual[year] += monthly_ttc

leasing_year_values = [leasing_annual[y] for y in YEAR_NUMS]

if "Redevances crédit-bail" in fixed_df["Rubrique"].astype(str).tolist():
    fixed_df.loc[fixed_df["Rubrique"].astype(str) == "Redevances crédit-bail", YEAR_LABELS] = leasing_year_values
else:
    new_row = {"Rubrique": "Redevances crédit-bail"}
    for idx, y in enumerate(YEAR_LABELS):
        new_row[y] = leasing_year_values[idx]
    fixed_df = pd.concat([fixed_df, pd.DataFrame([new_row])], ignore_index=True)

salary_detail_rows = []
salary_year_totals_gross = [0.0] * 5
salary_year_totals_social = [0.0] * 5
salary_year_totals_total = [0.0] * 5

for _, r in salary_df.iterrows():
    row = {"Rubrique": str(r["Poste"])}
    brut_mensuel = n(r["Salaire brut mensuel"])

    for idx, y in enumerate(YEAR_LABELS):
        headcount = n(r[y])
        gross_year = brut_mensuel * headcount * 12
        if idx == 0:
            gross_year *= (months_activity_y1 / 12)
        social_year = gross_year * employer_social_rate
        total_year = gross_year + social_year

        row[y] = total_year
        salary_year_totals_gross[idx] += gross_year
        salary_year_totals_social[idx] += social_year
        salary_year_totals_total[idx] += total_year

    salary_detail_rows.append(row)

df_salary_detail = pd.DataFrame(salary_detail_rows)
df_salary_summary = to_year_columns_df({
    "Salaires bruts": salary_year_totals_gross,
    "Charges sociales": salary_year_totals_social,
    "Total salaires + charges": salary_year_totals_total,
}, first_col_name="Rubrique")

fixed_total_years = [fixed_df[y].sum() for y in YEAR_LABELS]

ca_services_df_calc = ca_services_df.copy()
if not ca_services_df_calc.empty:
    ca_services_df_calc["CA brut annuel"] = (
        ca_services_df_calc["Nombre de jours ouvrés par an"] *
        ca_services_df_calc["Tarif par consultation"]
    )
    ca_services_df_calc["CA part clinique"] = (
        ca_services_df_calc["CA brut annuel"] *
        ca_services_df_calc["Part clinique (%)"] / 100
    )
    ca_services_df_calc["CA part médecin"] = (
        ca_services_df_calc["CA brut annuel"] *
        ca_services_df_calc["Part médecin (%)"] / 100
    )
else:
    ca_services_df_calc["CA brut annuel"] = []
    ca_services_df_calc["CA part clinique"] = []
    ca_services_df_calc["CA part médecin"] = []

ca_years_goods = [0.0] * 5
ca_years_services = [ca_services_df_calc["CA part clinique"].sum()] * 5
ca_total_years = [ca_services_df_calc["CA brut annuel"].sum()] * 5

df_ca_detail = pd.DataFrame([
    {
        "Rubrique": "CA clinique",
        "Année 1": ca_years_services[0],
        "Année 2": ca_years_services[1],
        "Année 3": ca_years_services[2],
        "Année 4": ca_years_services[3],
        "Année 5": ca_years_services[4],
    },
    {
        "Rubrique": "CA total",
        "Année 1": ca_total_years[0],
        "Année 2": ca_total_years[1],
        "Année 3": ca_total_years[2],
        "Année 4": ca_total_years[3],
        "Année 5": ca_total_years[4],
    },
])

df_ca_summary = to_year_columns_df({
    "CA marchandises": ca_years_goods,
    "CA services": ca_years_services,
    "CA total": ca_total_years,
}, first_col_name="Rubrique")

goods_purchases = [v * goods_purchase_rate for v in ca_years_goods]
medical_consumables = [v * medical_consumables_rate for v in ca_years_services]
office_supplies = [v * office_supplies_rate for v in ca_total_years]
other_variable = [v * other_variable_rate for v in ca_total_years]
waste_processing = [waste_kg_month * waste_cost_per_kg * 12 for _ in YEAR_NUMS]

variable_total = []
for idx in range(5):
    variable_total.append(
        goods_purchases[idx]
        + medical_consumables[idx]
        + office_supplies[idx]
        + other_variable[idx]
        + waste_processing[idx]
    )

df_variable_summary = to_year_columns_df({
    "Achats revendus marchandises": goods_purchases,
    "Consommables médicaux": medical_consumables,
    "Fournitures bureau variables": office_supplies,
    "Autres variables": other_variable,
    "Traitement déchets médicaux": waste_processing,
    "Total charges variables": variable_total,
}, first_col_name="Rubrique")

tax_map = {}
for _, r in taxes_df.iterrows():
    tax_map[str(r["Rubrique"])] = [n(r[y]) for y in YEAR_LABELS]

is_rates = tax_map.get("IS", [0.20] * 5)
tsc_rates = tax_map.get("Taxe services communaux", [0.105] * 5)
prof_tax_rates = tax_map.get("Taxe professionnelle", [0.20] * 5)
stamp_rates = tax_map.get("Droits d'enregistrement / timbre", [0.0025] * 5)
sign_tax_values = tax_map.get("Taxe enseigne", [200000.0] * 5)
non_current_rates = tax_map.get("Charges non courantes (% du CA)", [0.001] * 5)

locative_base_years = growth_series(locative_base_y1, [0.0, 0.0, 0.0, 0.0])
tsc_values = [locative_base_years[i] * tsc_rates[i] for i in range(5)]
if auto_exempt_prof_tax:
    prof_tax_values = [0.0] * 5
else:
    prof_tax_values = [locative_base_years[i] * prof_tax_rates[i] for i in range(5)]

cash_collected_base = [ca_total_years[i] * cash_share for i in range(5)]
stamp_values = [cash_collected_base[i] * stamp_rates[i] for i in range(5)]
non_current_values = [ca_total_years[i] * non_current_rates[i] for i in range(5)]

df_taxes_summary = to_year_columns_df({
    "Taxe services communaux": tsc_values,
    "Taxe professionnelle": prof_tax_values,
    "Droits d'enregistrement / timbre": stamp_values,
    "Taxe enseigne": sign_tax_values,
    "Charges non courantes": non_current_values,
}, first_col_name="Rubrique")

df_charge_summary = to_year_columns_df({
    "Salaires + charges": salary_year_totals_total,
    "Charges fixes hors salaires": fixed_total_years,
    "Taxes et redevances": [tsc_values[i] + prof_tax_values[i] + stamp_values[i] + sign_tax_values[i] for i in range(5)],
    "Charges non courantes": non_current_values,
    "Charges variables": variable_total,
    "Amortissements": amort_totals,
    "Charges financières": [n(v) for v in df_loans["Intérêts"].tolist()] if not df_loans.empty else [0.0] * 5,
    "Total charges": [
        variable_total[i]
        + fixed_total_years[i]
        + salary_year_totals_total[i]
        + tsc_values[i]
        + prof_tax_values[i]
        + stamp_values[i]
        + sign_tax_values[i]
        + non_current_values[i]
        + amort_totals[i]
        + (df_loans["Intérêts"].tolist()[i] if not df_loans.empty else 0.0)
        for i in range(5)
    ],
}, first_col_name="Rubrique")

bfr_days_map = {}
for _, r in bfr_days_df.iterrows():
    bfr_days_map[str(r["Rubrique"])] = [n(r[y]) for y in YEAR_LABELS]

stock_days = bfr_days_map.get("Stock (jours d'achats)", [15] * 5)
amo_days = bfr_days_map.get("Créances AMO (jours)", [120] * 5)
priv_days = bfr_days_map.get("Créances privées (jours)", [120] * 5)
uncov_days = bfr_days_map.get("Créances sans couverture (jours)", [0] * 5)
other_recv_days = bfr_days_map.get("Autres créances (jours)", [30] * 5)
supplier_days = bfr_days_map.get("Dettes fournisseurs (jours)", [60] * 5)
other_pay_days = bfr_days_map.get("Autres dettes (jours)", [30] * 5)

bfr_values = []
delta_bfr = []
for idx in range(5):
    purchases_base = goods_purchases[idx] + medical_consumables[idx] + office_supplies[idx] + other_variable[idx]
    stock_component = purchases_base * stock_days[idx] / 365
    amo_component = ca_total_years[idx] * amo_share * amo_days[idx] / 365
    priv_component = ca_total_years[idx] * private_share * priv_days[idx] / 365
    uncov_component = ca_total_years[idx] * uninsured_share * uncov_days[idx] / 365
    other_recv_component = ca_total_years[idx] * other_recv_days[idx] / 365
    supplier_component = purchases_base * supplier_days[idx] / 365
    other_pay_component = purchases_base * other_pay_days[idx] / 365

    bfr = stock_component + amo_component + priv_component + uncov_component + other_recv_component - supplier_component - other_pay_component
    bfr_values.append(bfr)

for idx in range(5):
    if idx == 0:
        delta_bfr.append(bfr_values[0])
    else:
        delta_bfr.append(bfr_values[idx] - bfr_values[idx - 1])

startup_purchases = (variable_total[0] / 12) * 3
startup_rent = 0.0
if "Loyer et charges locatives" in fixed_df["Rubrique"].astype(str).tolist():
    startup_rent = fixed_df.loc[fixed_df["Rubrique"].astype(str) == "Loyer et charges locatives", "Année 1"].sum() / 12 * 3
startup_payroll = salary_year_totals_total[0] / max(months_activity_y1, 1) * 3
startup_leasing = leasing_year_values[0] / 12
bfr_startup_total = startup_purchases + startup_rent + startup_payroll + startup_leasing

df_bfr_startup = pd.DataFrame([
    {"Rubrique": "Achats (3 mois)", "Montant": startup_purchases},
    {"Rubrique": "Loyers (3 mois)", "Montant": startup_rent},
    {"Rubrique": "Salaires et charges (3 mois)", "Montant": startup_payroll},
    {"Rubrique": "Redevances crédit-bail (1 mois)", "Montant": startup_leasing},
    {"Rubrique": "TOTAL BFR DE DÉMARRAGE", "Montant": bfr_startup_total},
])

df_bfr = to_year_columns_df({
    "BFR exploitation": bfr_values,
    "Variation BFR": delta_bfr,
}, first_col_name="Rubrique")

financial_interest = df_loans["Intérêts"].tolist() if not df_loans.empty else [0.0] * 5
financial_principal = df_loans["Remboursement capital"].tolist() if not df_loans.empty else [0.0] * 5
annual_debt_service = df_loans["Annuité"].tolist() if not df_loans.empty else [0.0] * 5

gross_margin = [ca_total_years[i] - variable_total[i] for i in range(5)]
ebitda = [gross_margin[i] - (fixed_total_years[i] + salary_year_totals_total[i] + tsc_values[i] + prof_tax_values[i] + stamp_values[i] + sign_tax_values[i] + non_current_values[i]) for i in range(5)]
ebit = [ebitda[i] - amort_totals[i] for i in range(5)]
pre_tax = [ebit[i] - financial_interest[i] for i in range(5)]
solidarity_values = [max(pre_tax[i], 0.0) * solidarity_rate(pre_tax[i]) for i in range(5)]
is_values = [max(pre_tax[i], 0.0) * is_rates[i] for i in range(5)]
net_income = [pre_tax[i] - is_values[i] - solidarity_values[i] for i in range(5)]
caf_values = [net_income[i] + amort_totals[i] for i in range(5)]

excess_or_gap = total_resources - (total_startup_needs + bfr_startup_total)
initial_cash_balance = cash_start_need + max(excess_or_gap, 0.0)

net_cash_flow = []
ending_cash = []
cash_balance = initial_cash_balance

for iyr in range(5):
    invest_out = capex_total_excl_cash if iyr == 0 else 0.0
    start_bfr_out = bfr_startup_total if iyr == 0 else 0.0
    financing_in = total_resources if iyr == 0 else 0.0

    flow = caf_values[iyr] - delta_bfr[iyr] - financial_principal[iyr] - invest_out - start_bfr_out + financing_in - solidarity_values[iyr]
    cash_balance += flow
    net_cash_flow.append(flow)
    ending_cash.append(cash_balance)

df_pnl = to_year_columns_df({
    "CA marchandises": ca_years_goods,
    "CA services": ca_years_services,
    "CA total": ca_total_years,
    "Charges variables": variable_total,
    "Marge brute": gross_margin,
    "Taux marge brute": [safe_div(gross_margin[i], ca_total_years[i]) for i in range(5)],
    "Charges fixes hors amort.": [fixed_total_years[i] + salary_year_totals_total[i] + tsc_values[i] + prof_tax_values[i] + stamp_values[i] + sign_tax_values[i] + non_current_values[i] for i in range(5)],
    "EBITDA": ebitda,
    "Marge EBITDA": [safe_div(ebitda[i], ca_total_years[i]) for i in range(5)],
    "Amortissements": amort_totals,
    "EBIT": ebit,
    "Charges financières": financial_interest,
    "Résultat avant impôt": pre_tax,
    "IS": is_values,
    "Taxe de solidarité": solidarity_values,
    "Résultat net": net_income,
    "Marge nette": [safe_div(net_income[i], ca_total_years[i]) for i in range(5)],
}, first_col_name="Rubrique")

df_cashflow = to_year_columns_df({
    "CAF": caf_values,
    "Variation BFR": delta_bfr,
    "Remboursement capital": financial_principal,
    "Investissements initiaux": [capex_total_excl_cash, 0, 0, 0, 0],
    "BFR de démarrage": [bfr_startup_total, 0, 0, 0, 0],
    "Ressources initiales": [total_resources, 0, 0, 0, 0],
    "Flux net de trésorerie": net_cash_flow,
    "Trésorerie fin d'année": ending_cash,
}, first_col_name="Rubrique")

df_funding_plan = to_year_columns_df({
    "Investissements": [capex_total_excl_cash, 0, 0, 0, 0],
    "Trésorerie de départ": [cash_start_need, 0, 0, 0, 0],
    "BFR de démarrage": [bfr_startup_total, 0, 0, 0, 0],
    "BFR exploitation": bfr_values,
    "Variation BFR": delta_bfr,
    "CAF": caf_values,
    "Remboursement capital": financial_principal,
    "Flux net": net_cash_flow,
    "Trésorerie fin d'année": ending_cash,
}, first_col_name="Rubrique")

df_funding_structure = pd.DataFrame([
    {"Rubrique": "Total besoins de démarrage", "Montant": total_startup_needs},
    {"Rubrique": "BFR de démarrage", "Montant": bfr_startup_total},
    {"Rubrique": "TOTAL BESOINS", "Montant": total_startup_needs + bfr_startup_total},
    {"Rubrique": "TOTAL RESSOURCES", "Montant": total_resources},
    {"Rubrique": "Excédent / Déficit", "Montant": excess_or_gap},
])

fixed_for_break_even = [fixed_total_years[i] + salary_year_totals_total[i] + tsc_values[i] + prof_tax_values[i] + stamp_values[i] + sign_tax_values[i] + non_current_values[i] for i in range(5)]
tmcv = [max(1 - safe_div(variable_total[i], max(ca_total_years[i], 1.0)), 0.0001) for i in range(5)]
break_even_values = [fixed_for_break_even[i] / tmcv[i] for i in range(5)]
safety_margin_values = [ca_total_years[i] - break_even_values[i] for i in range(5)]
safety_margin_pct = [safe_div(safety_margin_values[i], ca_total_years[i]) for i in range(5)]

df_break_even = to_year_columns_df({
    "Charges fixes pour SR": fixed_for_break_even,
    "TMCV": tmcv,
    "Seuil de rentabilité": break_even_values,
    "Marge de sécurité": safety_margin_values,
    "Marge de sécurité %": safety_margin_pct,
}, first_col_name="Rubrique")

value_added = [
    gross_margin[i] - (
        fixed_df.loc[fixed_df["Rubrique"].isin(["Téléphone / internet", "Autres abonnements", "Fournitures diverses"]), YEAR_LABELS[i]].sum()
        if not fixed_df.empty else 0.0
    )
    for i in range(5)
]

df_sig = to_year_columns_df({
    "Marge commerciale": [ca_years_goods[i] - goods_purchases[i] for i in range(5)],
    "Production de l'exercice": ca_years_services,
    "Valeur ajoutée": value_added,
    "EBE": ebitda,
    "Résultat d'exploitation": ebit,
    "Résultat courant": pre_tax,
    "Résultat net": net_income,
}, first_col_name="Rubrique")

goods_y1 = ca_years_goods[0]
services_y1 = ca_years_services[0]

dist_goods = normalize_percent_list(dist_df["% CA marchandises"].tolist())
dist_services = normalize_percent_list(dist_df["% CA services"].tolist())

monthly_goods = goods_y1 * dist_goods
monthly_services = services_y1 * dist_services
monthly_total_sales = monthly_goods + monthly_services

monthly_variable = (
    monthly_goods * goods_purchase_rate
    + monthly_services * medical_consumables_rate
    + monthly_total_sales * office_supplies_rate
    + monthly_total_sales * other_variable_rate
    + np.array([waste_kg_month * waste_cost_per_kg] * 12)
)

monthly_fixed_disbursed = (
    (fixed_total_years[0] + salary_year_totals_total[0] + tsc_values[0] + prof_tax_values[0] + stamp_values[0] + sign_tax_values[0] + non_current_values[0]) / 12
)

monthly_interest_y1 = financial_interest[0] / 12 if len(financial_interest) > 0 else 0.0
monthly_principal_y1 = financial_principal[0] / 12 if len(financial_principal) > 0 else 0.0
monthly_amort_y1 = amort_totals[0] / 12

client_delay_days = amo_days[0]
supplier_delay_days = supplier_days[0]

client_delay_months = int(round(client_delay_days / 30))
supplier_delay_months = int(round(supplier_delay_days / 30))

treasury_rows = []
cash_month = initial_cash_balance

for im, month in enumerate(MONTH_LABELS):
    collected = monthly_total_sales[im] if client_delay_months == 0 else (
        monthly_total_sales[im - client_delay_months] if im - client_delay_months >= 0 else 0.0
    )
    paid_var = monthly_variable[im] if supplier_delay_months == 0 else (
        monthly_variable[im - supplier_delay_months] if im - supplier_delay_months >= 0 else 0.0
    )

    net_m = collected - paid_var - monthly_fixed_disbursed - monthly_interest_y1 - monthly_principal_y1
    cash_month += net_m

    treasury_rows.append({
        "Mois": month,
        "Encaissements": collected,
        "Charges variables décaissées": paid_var,
        "Charges fixes décaissées": monthly_fixed_disbursed,
        "Intérêts": monthly_interest_y1,
        "Remboursement capital": monthly_principal_y1,
        "Amortissements non décaissés": monthly_amort_y1,
        "Flux net mensuel": net_m,
        "Trésorerie fin de mois": cash_month,
    })

df_monthly_treasury = pd.DataFrame(treasury_rows)

diagnostic = []

if net_income[0] > 0:
    diagnostic.append(("good", f"Le projet est bénéficiaire dès l'année 1 avec un résultat net de {fmt_mad(net_income[0])}."))
elif len(net_income) > 1 and net_income[1] > 0:
    diagnostic.append(("warn", f"Le projet devient bénéficiaire en année 2 avec un résultat net de {fmt_mad(net_income[1])}."))
else:
    diagnostic.append(("risk", "Le projet reste déficitaire au démarrage. Il faut revoir le chiffre d'affaires, les charges ou la structure de financement."))

if excess_or_gap >= 0:
    diagnostic.append(("good", f"Le financement couvre les besoins initiaux avec un excédent de {fmt_mad(excess_or_gap)}."))
else:
    diagnostic.append(("risk", f"Le financement initial présente un déficit de {fmt_mad(abs(excess_or_gap))}."))

if df_monthly_treasury["Trésorerie fin de mois"].min() < 0:
    diagnostic.append(("risk", f"La trésorerie mensuelle passe en négatif avec un point bas de {fmt_mad(df_monthly_treasury['Trésorerie fin de mois'].min())}."))
else:
    diagnostic.append(("good", "La trésorerie mensuelle reste positive sur l'année 1."))

if safety_margin_pct[0] >= 0.15:
    diagnostic.append(("good", f"La marge de sécurité de l'année 1 ressort à {fmt_pct(safety_margin_pct[0])}, ce qui est confortable."))
elif safety_margin_pct[0] >= 0.05:
    diagnostic.append(("warn", f"La marge de sécurité de l'année 1 est correcte mais à surveiller : {fmt_pct(safety_margin_pct[0])}."))
else:
    diagnostic.append(("risk", f"La marge de sécurité de l'année 1 est faible : {fmt_pct(safety_margin_pct[0])}."))

if annual_debt_service[0] > 0 and safe_div(annual_debt_service[0], ca_total_years[0]) > 0.20:
    diagnostic.append(("risk", "Le service de la dette pèse lourdement sur le CA année 1."))
elif annual_debt_service[0] > 0 and safe_div(annual_debt_service[0], ca_total_years[0]) > 0.10:
    diagnostic.append(("warn", "Le service de la dette est significatif et doit être suivi de près."))
else:
    diagnostic.append(("good", "Le poids annuel de la dette reste maîtrisable par rapport au chiffre d'affaires."))

df_invest_summary = pd.DataFrame([
    {"Rubrique": "Investissements hors trésorerie de départ", "Montant": capex_total_excl_cash},
    {"Rubrique": "Trésorerie de départ", "Montant": cash_start_need},
    {"Rubrique": "TOTAL besoins de démarrage", "Montant": total_startup_needs},
])

df_financing_sources = financing_df[["Source", "Type", "Montant"]].copy()

df_formula_reference = pd.DataFrame([
    {"Rubrique": "CA total", "Formule / logique": "CA marchandises + CA services", "Variables utilisées": "ca_years_goods + ca_years_services", "Commentaire": "Chiffre d'affaires global annuel"},
    {"Rubrique": "Charges variables", "Formule / logique": "Achats revendus + consommables médicaux + fournitures bureau variables + autres variables + déchets médicaux", "Variables utilisées": "goods_purchases + medical_consumables + office_supplies + other_variable + waste_processing", "Commentaire": "Charges directement liées au niveau d'activité"},
    {"Rubrique": "Marge brute", "Formule / logique": "CA total - Charges variables", "Variables utilisées": "ca_total_years - variable_total", "Commentaire": "Marge après charges variables"},
    {"Rubrique": "Taux marge brute", "Formule / logique": "Marge brute / CA total", "Variables utilisées": "gross_margin / ca_total_years", "Commentaire": "Indicateur de profitabilité brute"},
    {"Rubrique": "Salaires bruts annuels", "Formule / logique": "Salaire brut mensuel × effectif × 12 (prorata Année 1 selon mois d'activité)", "Variables utilisées": "brut_mensuel × effectif × 12 × prorata", "Commentaire": "Base du coût salarial"},
    {"Rubrique": "Charges sociales", "Formule / logique": "Salaires bruts annuels × taux charges sociales employeur", "Variables utilisées": "gross_year × employer_social_rate", "Commentaire": "Charges patronales"},
    {"Rubrique": "Total salaires + charges", "Formule / logique": "Salaires bruts + charges sociales", "Variables utilisées": "salary_year_totals_gross + salary_year_totals_social", "Commentaire": "Coût complet RH"},
    {"Rubrique": "Redevances crédit-bail", "Formule / logique": "Somme des mensualités TTC des contrats actifs sur chaque année", "Variables utilisées": "leasing_df", "Commentaire": "Automatiquement injecté dans les charges fixes"},
    {"Rubrique": "Amortissements", "Formule / logique": "Montant / durée d'amortissement", "Variables utilisées": "linear_amortization(amount, duration)", "Commentaire": "Répartition linéaire des investissements"},
    {"Rubrique": "BFR exploitation", "Formule / logique": "Stock + créances clients + autres créances - dettes fournisseurs - autres dettes", "Variables utilisées": "stock_component + amo_component + priv_component + uncov_component + other_recv_component - supplier_component - other_pay_component", "Commentaire": "Besoin structurel lié à l'exploitation"},
    {"Rubrique": "Variation BFR", "Formule / logique": "BFR année N - BFR année N-1", "Variables utilisées": "delta_bfr", "Commentaire": "Impact sur la trésorerie"},
    {"Rubrique": "BFR de démarrage", "Formule / logique": "Achats 3 mois + loyers 3 mois + salaires & charges 3 mois + crédit-bail 1 mois", "Variables utilisées": "startup_purchases + startup_rent + startup_payroll + startup_leasing", "Commentaire": "Besoin initial avant démarrage"},
    {"Rubrique": "EBITDA", "Formule / logique": "Marge brute - charges fixes hors amortissements", "Variables utilisées": "gross_margin - (fixed_total_years + salary_year_totals_total + taxes + charges non courantes)", "Commentaire": "Rentabilité opérationnelle avant amortissements"},
    {"Rubrique": "Marge EBITDA", "Formule / logique": "EBITDA / CA total", "Variables utilisées": "ebitda / ca_total_years", "Commentaire": "Taux de rentabilité opérationnelle"},
    {"Rubrique": "EBIT", "Formule / logique": "EBITDA - amortissements", "Variables utilisées": "ebitda - amort_totals", "Commentaire": "Résultat d'exploitation"},
    {"Rubrique": "Résultat avant impôt", "Formule / logique": "EBIT - charges financières", "Variables utilisées": "ebit - financial_interest", "Commentaire": "Résultat avant fiscalité"},
    {"Rubrique": "IS", "Formule / logique": "max(Résultat avant impôt, 0) × taux IS", "Variables utilisées": "max(pre_tax, 0) × is_rates", "Commentaire": "Calcul d'impôt société"},
    {"Rubrique": "Taxe de solidarité", "Formule / logique": "max(Résultat avant impôt, 0) × taux progressif selon palier", "Variables utilisées": "solidarity_rate(pre_tax)", "Commentaire": "Calcul selon le palier du résultat"},
    {"Rubrique": "Résultat net", "Formule / logique": "Résultat avant impôt - IS - taxe de solidarité", "Variables utilisées": "pre_tax - is_values - solidarity_values", "Commentaire": "Résultat final après impôts"},
    {"Rubrique": "Marge nette", "Formule / logique": "Résultat net / CA total", "Variables utilisées": "net_income / ca_total_years", "Commentaire": "Rentabilité nette"},
    {"Rubrique": "CAF", "Formule / logique": "Résultat net + amortissements", "Variables utilisées": "net_income + amort_totals", "Commentaire": "Capacité d'autofinancement"},
    {"Rubrique": "Flux net de trésorerie", "Formule / logique": "CAF - variation BFR - remboursement capital - investissements - BFR démarrage + ressources initiales - taxe de solidarité", "Variables utilisées": "caf_values - delta_bfr - financial_principal - invest_out - start_bfr_out + financing_in - solidarity_values", "Commentaire": "Flux annuel de cash"},
    {"Rubrique": "Trésorerie fin d'année", "Formule / logique": "Trésorerie début + flux net de trésorerie", "Variables utilisées": "cash_balance + flow", "Commentaire": "Position de trésorerie annuelle"},
    {"Rubrique": "TMCV", "Formule / logique": "1 - (Charges variables / CA total)", "Variables utilisées": "1 - variable_total / ca_total_years", "Commentaire": "Taux de marge sur coûts variables"},
    {"Rubrique": "Seuil de rentabilité", "Formule / logique": "Charges fixes pour SR / TMCV", "Variables utilisées": "fixed_for_break_even / tmcv", "Commentaire": "Niveau de CA minimum pour couvrir les charges"},
    {"Rubrique": "Marge de sécurité", "Formule / logique": "CA total - seuil de rentabilité", "Variables utilisées": "ca_total_years - break_even_values", "Commentaire": "Distance par rapport au point mort"},
    {"Rubrique": "Marge de sécurité %", "Formule / logique": "Marge de sécurité / CA total", "Variables utilisées": "safety_margin_values / ca_total_years", "Commentaire": "Sécurité économique du projet"},
])

# =========================================================
# PRINT TAB
# =========================================================
with tab_print:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Besoins démarrage", fmt_mad(total_startup_needs))
    m2.metric("BFR démarrage", fmt_mad(bfr_startup_total))
    m3.metric("CA année 1", fmt_mad(ca_total_years[0]))
    m4.metric("Résultat net année 1", fmt_mad(net_income[0]))
    m5.metric("Trésorerie fin année 5", fmt_mad(ending_cash[4]))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Investissements et financements</div>', unsafe_allow_html=True)
        st.dataframe(money_style(df_invest_summary, non_money_cols=["Rubrique"]), use_container_width=True)
        st.dataframe(money_style(df_financing_sources, non_money_cols=["Source", "Type"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Structure de financement du projet</div>', unsafe_allow_html=True)
        st.dataframe(money_style(df_funding_structure, non_money_cols=["Rubrique"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Plan de financement sur 5 ans</div>', unsafe_allow_html=True)
        st.dataframe(money_style(df_funding_plan, non_money_cols=["Rubrique"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Seuil de rentabilité économique</div>', unsafe_allow_html=True)
        st.dataframe(money_style(df_break_even, non_money_cols=["Rubrique"]), use_container_width=True)
        st.markdown("<p class='sub-note'>Les lignes TMCV et Marge de sécurité % sont à interpréter comme des pourcentages.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Salaires et charges sociales</div>', unsafe_allow_html=True)
        st.dataframe(money_style(df_salary_summary, non_money_cols=["Rubrique"]), use_container_width=True)
        st.markdown("<p class='sub-note'>Détail par postes</p>", unsafe_allow_html=True)
        st.dataframe(money_style(df_salary_detail, non_money_cols=["Rubrique"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col6:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Détail des amortissements</div>', unsafe_allow_html=True)
        st.dataframe(money_style(df_amort, non_money_cols=["Rubrique", "Durée amort. (ans)"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col7, col8 = st.columns(2)
    with col7:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Synthèse des charges</div>', unsafe_allow_html=True)
        st.dataframe(money_style(df_charge_summary, non_money_cols=["Rubrique"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col8:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Synthèse du chiffre d’affaires</div>', unsafe_allow_html=True)
        st.dataframe(money_style(df_ca_summary, non_money_cols=["Rubrique"]), use_container_width=True)
        st.markdown("<p class='sub-note'>Les montants proviennent de la rubrique consultations / parts clinique-médecin saisie dans Données à saisir.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col9, col10 = st.columns(2)
    with col9:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Compte de résultat prévisionnel sur 5 ans</div>', unsafe_allow_html=True)
        st.dataframe(money_style(df_pnl, non_money_cols=["Rubrique"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col10:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Soldes intermédiaires de gestion</div>', unsafe_allow_html=True)
        st.dataframe(money_style(df_sig, non_money_cols=["Rubrique"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col11, col12 = st.columns(2)
    with col11:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">BFR de démarrage</div>', unsafe_allow_html=True)
        st.dataframe(money_style(df_bfr_startup, non_money_cols=["Rubrique"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col12:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">BFR d’exploitation</div>', unsafe_allow_html=True)
        st.dataframe(money_style(df_bfr, non_money_cols=["Rubrique"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col13, col14 = st.columns(2)
    with col13:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Budget prévisionnel de trésorerie - année 1</div>', unsafe_allow_html=True)
        st.dataframe(money_style(df_monthly_treasury, non_money_cols=["Mois"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col14:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Cash-flow du projet</div>', unsafe_allow_html=True)
        st.dataframe(money_style(df_cashflow, non_money_cols=["Rubrique"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Tableau des emprunts</div>', unsafe_allow_html=True)
    if not df_loans.empty:
        st.dataframe(money_style(df_loans, non_money_cols=["Année"]), use_container_width=True)
    else:
        st.info("Aucun emprunt renseigné.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# REPORT TAB
# =========================================================
with tab_report:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Résumé exécutif</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
**Projet analysé :** {project_name}  
**Porteur :** {project_holder}  
**Ville :** {city}  
**Statut juridique :** {legal_status}  
**Horizon étudié :** 5 ans  

- Total besoins de démarrage : **{fmt_mad(total_startup_needs)}**
- BFR de démarrage : **{fmt_mad(bfr_startup_total)}**
- Ressources initiales : **{fmt_mad(total_resources)}**
- CA année 1 : **{fmt_mad(ca_total_years[0])}**
- CA année 5 : **{fmt_mad(ca_total_years[4])}**
- Résultat net année 1 : **{fmt_mad(net_income[0])}**
- Résultat net année 5 : **{fmt_mad(net_income[4])}**
- Trésorerie fin année 5 : **{fmt_mad(ending_cash[4])}**
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Contrôle global automatique</div>', unsafe_allow_html=True)
    for level, msg in diagnostic:
        css = "good-box" if level == "good" else "warn-box" if level == "warn" else "risk-box"
        st.markdown(f"<div class='{css}'>{msg}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    recommendations = []
    if excess_or_gap < 0:
        recommendations.append(f"Compléter les ressources initiales d’environ {fmt_mad(abs(excess_or_gap))}.")
    if df_monthly_treasury["Trésorerie fin de mois"].min() < 0:
        recommendations.append("Prévoir plus de trésorerie de départ ou négocier un différé d’emprunt.")
    if safety_margin_pct[0] < 0.10:
        recommendations.append("Sécuriser un volume minimal d’activité pour éloigner le seuil de rentabilité.")
    if safe_div(annual_debt_service[0], max(ca_total_years[0], 1.0)) > 0.15:
        recommendations.append("Réduire le poids de la dette ou allonger la durée de remboursement.")
    if not recommendations:
        recommendations = [
            "Le montage financier paraît cohérent au regard des hypothèses saisies.",
            "Mettre à jour le prévisionnel chaque trimestre avec les réalisations effectives.",
            "Suivre mensuellement les encaissements, le BFR et la masse salariale.",
        ]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Recommandations</div>', unsafe_allow_html=True)
    for idx, reco in enumerate(recommendations, start=1):
        st.markdown(f"**{idx}.** {reco}")
    st.markdown('</div>', unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        chart_df = pd.DataFrame({"Année": YEAR_LABELS, "CA": ca_total_years, "Résultat net": net_income})
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=chart_df["Année"], y=chart_df["CA"], name="CA"))
        fig1.add_trace(go.Scatter(x=chart_df["Année"], y=chart_df["Résultat net"], mode="lines+markers", name="Résultat net"))
        fig1.update_layout(title="Évolution du CA et du résultat net", height=420)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig2 = px.line(
            pd.DataFrame({"Année": YEAR_LABELS, "Trésorerie": ending_cash}),
            x="Année",
            y="Trésorerie",
            markers=True,
            title="Évolution de la trésorerie",
        )
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    g3, g4 = st.columns(2)
    with g3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig3 = px.bar(
            pd.DataFrame({
                "Année": YEAR_LABELS,
                "BFR exploitation": bfr_values,
                "Variation BFR": delta_bfr
            }),
            x="Année",
            y=["BFR exploitation", "Variation BFR"],
            barmode="group",
            title="BFR et variation du BFR"
        )
        fig3.update_layout(height=420)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with g4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig4 = px.bar(
            pd.DataFrame({
                "Année": YEAR_LABELS,
                "Salaires + charges": salary_year_totals_total,
                "Charges fixes hors salaires": fixed_total_years,
                "Charges variables": variable_total
            }),
            x="Année",
            y=["Salaires + charges", "Charges fixes hors salaires", "Charges variables"],
            barmode="stack",
            title="Structure des charges"
        )
        fig4.update_layout(height=420)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# EXCEL EXPORT
# =========================================================
def write_block_to_sheet(
    worksheet,
    workbook,
    start_row,
    title,
    df,
    percent_rows=None,
    money_default=True,
):
    percent_rows = percent_rows or []

    fmt_title = workbook.add_format({
        "bold": True,
        "font_color": "white",
        "bg_color": "#0F4C81",
        "align": "center",
        "valign": "vcenter",
        "border": 1,
        "font_size": 12,
    })
    fmt_header = workbook.add_format({
        "bold": True,
        "font_color": "#0F2B57",
        "bg_color": "#D9EEF9",
        "align": "center",
        "valign": "vcenter",
        "border": 1,
    })
    fmt_text = workbook.add_format({
        "border": 1,
        "align": "left",
        "valign": "vcenter",
    })
    fmt_money = workbook.add_format({
        "border": 1,
        "align": "right",
        "valign": "vcenter",
        "num_format": "#,##0;[Red]-#,##0",
    })
    fmt_pct = workbook.add_format({
        "border": 1,
        "align": "right",
        "valign": "vcenter",
        "num_format": "0.0%",
    })

    ncols = len(df.columns)
    worksheet.merge_range(start_row, 0, start_row, ncols - 1, title, fmt_title)
    start_row += 1

    for c_idx, col in enumerate(df.columns):
        worksheet.write(start_row, c_idx, col, fmt_header)
    start_row += 1

    for r_idx in range(len(df)):
        for c_idx, col in enumerate(df.columns):
            val = df.iloc[r_idx, c_idx]
            row_label = str(df.iloc[r_idx, 0]) if len(df.columns) > 0 else ""

            if c_idx == 0:
                worksheet.write(start_row + r_idx, c_idx, val, fmt_text)
            else:
                if row_label in percent_rows:
                    worksheet.write_number(start_row + r_idx, c_idx, n(val), fmt_pct)
                elif money_default and isinstance(val, (int, float, np.integer, np.floating)):
                    worksheet.write_number(start_row + r_idx, c_idx, n(val), fmt_money)
                else:
                    worksheet.write(start_row + r_idx, c_idx, val, fmt_text)

    worksheet.set_column(0, 0, 38)
    for c_idx in range(1, ncols):
        worksheet.set_column(c_idx, c_idx, 16)

    return start_row + len(df) + 2


def make_excel_file():
    output = BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book

        cover_fmt = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#0F4C81",
            "font_size": 16,
            "align": "center",
            "valign": "vcenter",
            "border": 1,
        })
        sub_fmt = workbook.add_format({
            "bold": True,
            "font_color": "#0F2B57",
            "bg_color": "#D9EEF9",
            "align": "left",
            "border": 1,
        })
        text_fmt = workbook.add_format({"border": 1})
        money_fmt = workbook.add_format({"border": 1, "num_format": "#,##0;[Red]-#,##0"})

        ws_input = workbook.add_worksheet("Données à saisir")
        writer.sheets["Données à saisir"] = ws_input
        ws_input.merge_range("A1:F1", "DONNÉES À SAISIR", cover_fmt)
        ws_input.write("A3", "Projet", sub_fmt)
        ws_input.write("B3", project_name, text_fmt)
        ws_input.write("A4", "Porteur", sub_fmt)
        ws_input.write("B4", project_holder, text_fmt)
        ws_input.write("A5", "Ville", sub_fmt)
        ws_input.write("B5", city, text_fmt)
        ws_input.write("A6", "Statut juridique", sub_fmt)
        ws_input.write("B6", legal_status, text_fmt)
        ws_input.write("A7", "Nature activité", sub_fmt)
        ws_input.write("B7", activity_nature, text_fmt)

        row = 9
        for title, df in [
            ("Besoins de démarrage", investments_df),
            ("Financement", financing_df),
            ("Crédit-bail / Leasing", leasing_df),
            ("Salaires", salary_df),
            ("Charges fixes", fixed_df),
            ("Chiffre d'affaires consultations", ca_services_df),
            ("Répartition mensuelle", dist_df),
            ("BFR (jours)", bfr_days_df),
            ("Taxes et paramètres", taxes_df),
        ]:
            ws_input.write(row, 0, title, sub_fmt)
            row += 1
            df.to_excel(writer, sheet_name="Données à saisir", startrow=row, startcol=0, index=False)
            row += len(df) + 3

        ws_input.set_column("A:A", 32)
        ws_input.set_column("B:Z", 16)

        ws_plan = workbook.add_worksheet("Plan financier à imprimer")
        writer.sheets["Plan financier à imprimer"] = ws_plan
        ws_plan.merge_range("A1:F1", "ÉTUDE FINANCIÈRE PRÉVISIONNELLE SUR 5 ANS", cover_fmt)
        ws_plan.merge_range("A2:F2", project_name, cover_fmt)

        r = 4
        r = write_block_to_sheet(ws_plan, workbook, r, "Investissements et financements", df_invest_summary)
        r = write_block_to_sheet(ws_plan, workbook, r, "Structure de financement", df_funding_structure)
        r = write_block_to_sheet(ws_plan, workbook, r, "Plan de financement sur 5 ans", df_funding_plan)
        r = write_block_to_sheet(ws_plan, workbook, r, "Seuil de rentabilité", df_break_even, percent_rows=["TMCV", "Marge de sécurité %"])
        r = write_block_to_sheet(ws_plan, workbook, r, "Salaires et charges sociales", df_salary_summary)
        r = write_block_to_sheet(ws_plan, workbook, r, "Détail des amortissements", df_amort)
        r = write_block_to_sheet(ws_plan, workbook, r, "Synthèse des charges", df_charge_summary)
        r = write_block_to_sheet(ws_plan, workbook, r, "Synthèse du chiffre d'affaires", df_ca_summary)
        r = write_block_to_sheet(ws_plan, workbook, r, "Compte de résultat prévisionnel", df_pnl, percent_rows=["Taux marge brute", "Marge EBITDA", "Marge nette"])
        r = write_block_to_sheet(ws_plan, workbook, r, "Soldes intermédiaires de gestion", df_sig)
        r = write_block_to_sheet(ws_plan, workbook, r, "BFR de démarrage", df_bfr_startup)
        r = write_block_to_sheet(ws_plan, workbook, r, "BFR d'exploitation", df_bfr)
        r = write_block_to_sheet(ws_plan, workbook, r, "Cash-flow du projet", df_cashflow)

        if not df_loans.empty:
            loans_export = df_loans.copy()
            loans_export["Année"] = loans_export["Année"].astype(str)
            r = write_block_to_sheet(ws_plan, workbook, r, "Tableau des emprunts", loans_export, money_default=True)

        ws_calc = workbook.add_worksheet("Base de calculs")
        writer.sheets["Base de calculs"] = ws_calc
        ws_calc.merge_range("A1:D1", "DICTIONNAIRE DES FORMULES", cover_fmt)

        df_formula_reference.to_excel(
            writer,
            sheet_name="Base de calculs",
            startrow=2,
            startcol=0,
            index=False
        )

        ws_calc.set_column("A:A", 32)
        ws_calc.set_column("B:B", 42)
        ws_calc.set_column("C:C", 42)
        ws_calc.set_column("D:D", 40)

        ws_treso = workbook.add_worksheet("Trésorerie mensuelle")
        writer.sheets["Trésorerie mensuelle"] = ws_treso
        ws_treso.merge_range("A1:I1", "BUDGET PRÉVISIONNEL DE TRÉSORERIE - ANNÉE 1", cover_fmt)
        df_monthly_treasury.to_excel(writer, sheet_name="Trésorerie mensuelle", startrow=2, startcol=0, index=False)
        ws_treso.set_column("A:A", 16)
        ws_treso.set_column("B:I", 18, money_fmt)

    output.seek(0)
    return output

# =========================================================
# PDF EXPORT
# =========================================================
def add_pdf_table(elements, title, df, percent_rows=None):
    percent_rows = percent_rows or []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "tbl_title",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=colors.HexColor("#0F4C81"),
        spaceAfter=6,
        spaceBefore=6,
    )
    elements.append(Paragraph(title, title_style))

    data = [list(df.columns)]
    for _, row in df.iterrows():
        vals = []
        row_label = str(row.iloc[0]) if len(row) > 0 else ""
        for col, val in row.items():
            if isinstance(val, (int, float, np.integer, np.floating)):
                if row_label in percent_rows:
                    vals.append(f"{val:.1%}")
                else:
                    vals.append(f"{val:,.0f}".replace(",", " "))
            else:
                vals.append(str(val))
        data.append(vals)

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F4C81")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0DCEB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#EEF6FC")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.25 * cm))


def make_pdf_file():
    output = BytesIO()

    doc = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        rightMargin=0.8 * cm,
        leftMargin=0.8 * cm,
        topMargin=0.8 * cm,
        bottomMargin=0.8 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "pdf_title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=colors.HexColor("#0F2B57"),
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    sub_style = ParagraphStyle(
        "pdf_sub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#415A77"),
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    sec_style = ParagraphStyle(
        "pdf_sec",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor("#0F4C81"),
        spaceBefore=8,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "pdf_body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#16263D"),
        alignment=TA_LEFT,
        leading=13,
    )

    elems = []
    elems.append(Paragraph("ÉTUDE FINANCIÈRE PRÉVISIONNELLE SUR 5 ANS", title_style))
    elems.append(Paragraph(project_name, sub_style))
    elems.append(Paragraph(f"{project_holder} — {city} — Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}", sub_style))

    elems.append(Paragraph("Résumé exécutif", sec_style))
    elems.append(Paragraph(
        f"""
        Le projet mobilise {fmt_mad(total_startup_needs)} de besoins de démarrage et {fmt_mad(bfr_startup_total)} de BFR de démarrage.
        Les ressources initiales s’élèvent à {fmt_mad(total_resources)}.
        Le chiffre d’affaires passe de {fmt_mad(ca_total_years[0])} en année 1 à {fmt_mad(ca_total_years[4])} en année 5.
        Le résultat net évolue de {fmt_mad(net_income[0])} à {fmt_mad(net_income[4])}.
        La trésorerie de fin de période atteint {fmt_mad(ending_cash[4])}.
        """,
        body_style
    ))

    elems.append(Paragraph("Diagnostic", sec_style))
    for level, txt in diagnostic:
        prefix = "OK" if level == "good" else "Alerte" if level == "warn" else "Risque"
        elems.append(Paragraph(f"• <b>{prefix}</b> — {txt}", body_style))

    add_pdf_table(elems, "Investissements et financements", df_invest_summary)
    add_pdf_table(elems, "Structure de financement", df_funding_structure)

    elems.append(PageBreak())
    add_pdf_table(elems, "Compte de résultat prévisionnel", df_pnl, percent_rows=["Taux marge brute", "Marge EBITDA", "Marge nette"])
    add_pdf_table(elems, "Plan de financement sur 5 ans", df_funding_plan)
    add_pdf_table(elems, "Seuil de rentabilité", df_break_even, percent_rows=["TMCV", "Marge de sécurité %"])

    elems.append(PageBreak())
    add_pdf_table(elems, "Salaires et charges sociales", df_salary_summary)
    add_pdf_table(elems, "Détail des amortissements", df_amort)
    add_pdf_table(elems, "BFR de démarrage", df_bfr_startup)
    add_pdf_table(elems, "BFR d'exploitation", df_bfr)

    elems.append(PageBreak())
    add_pdf_table(elems, "Dictionnaire des formules", df_formula_reference)

    elems.append(PageBreak())
    add_pdf_table(elems, "Cash-flow du projet", df_cashflow)
    add_pdf_table(elems, "Budget prévisionnel de trésorerie - année 1", df_monthly_treasury)

    if not df_loans.empty:
        add_pdf_table(elems, "Tableau des emprunts", df_loans)

    doc.build(elems)
    output.seek(0)
    return output

# =========================================================
# CALC BASE TAB
# =========================================================
with tab_calc:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dictionnaire des formules</div>', unsafe_allow_html=True)
    st.dataframe(df_formula_reference, use_container_width=True, height=650)
    st.markdown(
        "<p class='sub-note'>Cette rubrique contient uniquement le dictionnaire des formules, méthodes de calcul, ratios, indicateurs et KPI utilisés dans l’outil.</p>",
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# EXPORT TAB
# =========================================================
with tab_export:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Exports professionnels</div>', unsafe_allow_html=True)
    st.write("L’Excel exporté garde une présentation structurée type cabinet, avec blocs, titres, années en colonnes et rubriques en lignes. Le PDF reprend la même logique en version imprimable.")

    excel_bytes = make_excel_file()
    pdf_bytes = make_pdf_file()

    st.download_button(
        "Télécharger l’étude financière en Excel",
        data=excel_bytes,
        file_name=f"{project_name.replace(' ', '_')}_etude_financiere.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.download_button(
        "Télécharger le rapport PDF",
        data=pdf_bytes,
        file_name=f"{project_name.replace(' ', '_')}_rapport_financier.pdf",
        mime="application/pdf",
    )

    st.markdown(
        """
        <p class='sub-note'>
        requirements.txt minimal :
        streamlit
        pandas
        numpy
        plotly
        xlsxwriter
        reportlab
        openpyxl
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div style="margin-top:18px; padding:14px; border-top:1px solid #dbe7f6; color:#4b6485;">
        Étude financière complète 5 ans — version Streamlit professionnelle
    </div>
    """,
    unsafe_allow_html=True,
)