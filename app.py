from io import BytesIO
from datetime import datetime

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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Étude financière & Business Plan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

YEARS = [1, 2, 3, 4, 5]
MONTHS = [
    "Mois 1", "Mois 2", "Mois 3", "Mois 4", "Mois 5", "Mois 6",
    "Mois 7", "Mois 8", "Mois 9", "Mois 10", "Mois 11", "Mois 12"
]
MONTHS_SHORT = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]

# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], .main {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%) !important;
        color: #f8fafc !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617 0%, #0f172a 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stHeader"] { background: rgba(0,0,0,0) !important; }
    .block-container { max-width: 1700px; padding-top: 1rem; padding-bottom: 2rem; }
    h1, h2, h3, h4, h5, h6, p, label, div, span, li { color: #f8fafc !important; }
    .hero-card {
        background: linear-gradient(135deg, rgba(37,99,235,0.35), rgba(16,185,129,0.18));
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 24px;
        padding: 24px;
        margin-bottom: 18px;
        box-shadow: 0 16px 40px rgba(0,0,0,0.22);
    }
    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }
    .section-title {
        font-size: 1.12rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    .small-note {
        font-size: 0.92rem;
        color: #cbd5e1 !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 18px !important;
        padding: 12px !important;
    }
    .good-box, .warn-box, .risk-box {
        padding: 14px; border-radius: 14px; margin-bottom: 10px;
        border: 1px solid rgba(255,255,255,0.10);
    }
    .good-box { background: rgba(16,185,129,0.18); border-color: rgba(16,185,129,0.35); }
    .warn-box { background: rgba(245,158,11,0.18); border-color: rgba(245,158,11,0.35); }
    .risk-box { background: rgba(239,68,68,0.18); border-color: rgba(239,68,68,0.35); }
    .legend-chip {
        display:inline-block;
        padding:4px 10px;
        border-radius:999px;
        font-size:0.82rem;
        background:rgba(255,255,255,0.08);
        border:1px solid rgba(255,255,255,0.10);
        margin-right:8px;
        margin-bottom:8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================
def fmt_mad(x: float) -> str:
    try:
        return f"{x:,.0f} MAD".replace(",", " ")
    except Exception:
        return "0 MAD"


def fmt_pct(x: float) -> str:
    try:
        return f"{x:.1%}"
    except Exception:
        return "0.0%"


def safe_div(a: float, b: float) -> float:
    if b in (0, None) or pd.isna(b):
        return 0.0
    return a / b


def annuity_payment(principal: float, annual_rate: float, years: int) -> float:
    if principal <= 0 or years <= 0:
        return 0.0
    monthly_rate = annual_rate / 12
    n = years * 12
    if monthly_rate == 0:
        return principal / n
    return principal * (monthly_rate / (1 - (1 + monthly_rate) ** (-n)))


def build_loan_schedule(principal: float, annual_rate: float, years: int, projection_years: int = 5) -> pd.DataFrame:
    rows = []
    if principal <= 0 or years <= 0:
        for y in range(1, projection_years + 1):
            rows.append({
                "Année": y,
                "Mensualité": 0.0,
                "Annuité": 0.0,
                "Intérêts": 0.0,
                "Remboursement capital": 0.0,
                "Capital restant dû": 0.0,
            })
        return pd.DataFrame(rows)

    monthly_payment = annuity_payment(principal, annual_rate, years)
    monthly_rate = annual_rate / 12
    balance = principal
    year_interest = 0.0
    year_principal = 0.0

    for month in range(1, projection_years * 12 + 1):
        if balance > 1e-8:
            interest = balance * monthly_rate
            principal_paid = min(monthly_payment - interest, balance)
            balance -= principal_paid
        else:
            interest = 0.0
            principal_paid = 0.0

        year_interest += interest
        year_principal += principal_paid

        if month % 12 == 0:
            year = month // 12
            rows.append({
                "Année": year,
                "Mensualité": monthly_payment,
                "Annuité": year_interest + year_principal,
                "Intérêts": year_interest,
                "Remboursement capital": year_principal,
                "Capital restant dû": max(balance, 0.0),
            })
            year_interest = 0.0
            year_principal = 0.0

    return pd.DataFrame(rows)


def linear_amortization(amount: float, duration_years: int, projection_years: int = 5) -> list:
    if amount <= 0 or duration_years <= 0:
        return [0.0] * projection_years
    annual = amount / duration_years
    return [annual if y <= duration_years else 0.0 for y in range(1, projection_years + 1)]


def normalize_distribution(values: list) -> np.ndarray:
    arr = np.array(values, dtype=float)
    total = arr.sum()
    if total <= 0:
        return np.array([1 / 12] * 12, dtype=float)
    return arr / total


def money_styler(df: pd.DataFrame, exclude=None):
    exclude = exclude or []
    fmt_map = {col: "{:,.0f}" for col in df.columns if col not in exclude}
    return df.style.format(fmt_map)


def pct_money_styler(df: pd.DataFrame, pct_cols=None, exclude=None):
    pct_cols = pct_cols or []
    exclude = exclude or []
    fmt_map = {}
    for col in df.columns:
        if col in exclude:
            continue
        fmt_map[col] = "{:.1%}" if col in pct_cols else "{:,.0f}"
    return df.style.format(fmt_map)


def add_pdf_table(elements, title, df, pct_cols=None):
    pct_cols = pct_cols or []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "pdf_table_title",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=colors.HexColor("#0F4C81"),
        spaceAfter=8,
        spaceBefore=6,
    )
    elements.append(Paragraph(title, title_style))

    data = [list(df.columns)]
    for _, row in df.iterrows():
        vals = []
        for col, item in row.items():
            if isinstance(item, (int, float, np.integer, np.floating)):
                vals.append(f"{item:.1%}" if col in pct_cols else f"{item:,.0f}".replace(",", " "))
            else:
                vals.append(str(item))
        data.append(vals)

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F4C81")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F8FAFC")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.3 * cm))


def make_excel_export(frames: dict) -> BytesIO:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book

        fmt_header = workbook.add_format({
            "bold": True, "font_color": "white", "bg_color": "#0F4C81",
            "align": "center", "valign": "vcenter", "border": 0
        })
        fmt_title = workbook.add_format({
            "bold": True, "font_size": 14, "font_color": "white", "bg_color": "#111827"
        })
        fmt_money = workbook.add_format({"num_format": "#,##0;[Red](#,##0)", "align": "right"})
        fmt_pct = workbook.add_format({"num_format": "0.0%", "align": "right"})
        fmt_text = workbook.add_format({"align": "left"})

        for sheet_name, payload in frames.items():
            df = payload["df"].copy()
            pct_cols = payload.get("pct_cols", [])
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, startrow=2, index=False)
            ws = writer.sheets[safe_name]
            ws.write(0, 0, sheet_name, fmt_title)
            ws.set_row(2, 24)

            for idx, col in enumerate(df.columns):
                ws.write(2, idx, col, fmt_header)
                width = max(13, min(28, len(str(col)) + 4))
                series = df[col]

                if col in pct_cols:
                    ws.set_column(idx, idx, width, fmt_pct)
                elif pd.api.types.is_numeric_dtype(series):
                    ws.set_column(idx, idx, width, fmt_money)
                else:
                    max_len = max([len(str(col))] + [len(str(v)) for v in series.head(100).fillna("")])
                    ws.set_column(idx, idx, min(max(max_len + 3, width), 40), fmt_text)

            ws.freeze_panes(3, 1)

    output.seek(0)
    return output


def make_pdf_report(project_name: str, sector: str, summary_text: str, diagnostics: list, tables: dict) -> BytesIO:
    output = BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        rightMargin=1.0 * cm,
        leftMargin=1.0 * cm,
        topMargin=1.0 * cm,
        bottomMargin=1.0 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title_style",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=8,
    )
    sub_style = ParagraphStyle(
        "sub_style",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#334155"),
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "body_style",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        alignment=TA_LEFT,
        leading=13,
        textColor=colors.HexColor("#111827"),
    )
    section_style = ParagraphStyle(
        "section_style",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor("#0F4C81"),
        spaceBefore=10,
        spaceAfter=8,
    )

    elems = []
    elems.append(Paragraph(f"Rapport d'étude financière - {project_name}", title_style))
    elems.append(Paragraph(f"Secteur : {sector} | Généré le {datetime.now().strftime('%d/%m/%Y %H:%M')}", sub_style))
    elems.append(Paragraph("Résumé exécutif", section_style))
    elems.append(Paragraph(summary_text, body_style))
    elems.append(Spacer(1, 0.15 * cm))

    elems.append(Paragraph("Diagnostic expert", section_style))
    for level, text in diagnostics:
        prefix = "OK" if level == "good" else "Alerte" if level == "warn" else "Risque"
        elems.append(Paragraph(f"• <b>{prefix}</b> - {text}", body_style))
    elems.append(Spacer(1, 0.2 * cm))

    for title, payload in tables.items():
        add_pdf_table(elems, title, payload["df"], pct_cols=payload.get("pct_cols", []))

    doc.build(elems)
    output.seek(0)
    return output


def generate_diagnostics(df_global, df_monthly_cash, funding_gap, break_even_df):
    comments = []
    y1 = df_global.iloc[0]
    y2 = df_global.iloc[1]
    y5 = df_global.iloc[-1]

    treso1 = fmt_mad(y1["Trésorerie fin d'année"])
    treso5 = fmt_mad(y5["Trésorerie fin d'année"])

    if y1["Résultat net"] > 0:
        comments.append(("good", f"Le projet est rentable dès l'année 1 avec un résultat net de {fmt_mad(y1['Résultat net'])}."))
    elif y2["Résultat net"] > 0:
        comments.append(("warn", f"Le projet devient bénéficiaire en année 2 avec un résultat net de {fmt_mad(y2['Résultat net'])}."))
    else:
        comments.append(("risk", "Le projet reste déficitaire sur les premières années. Il faut revoir le chiffre d'affaires, les charges ou la structure de financement."))

    if y1["Marge EBITDA %"] >= 0.15:
        comments.append(("good", f"La marge EBITDA de départ ressort à {fmt_pct(y1['Marge EBITDA %'])}, ce qui est solide."))
    elif y1["Marge EBITDA %"] >= 0.05:
        comments.append(("warn", f"La marge EBITDA de départ est positive mais encore modérée à {fmt_pct(y1['Marge EBITDA %'])}."))
    else:
        comments.append(("risk", f"La marge EBITDA de départ est faible à {fmt_pct(y1['Marge EBITDA %'])}."))

    if funding_gap >= 0:
        comments.append(("good", f"Le plan de financement couvre les besoins de démarrage avec un excédent de {fmt_mad(funding_gap)}."))
    else:
        comments.append(("risk", f"Le plan de financement présente un déficit initial de {fmt_mad(abs(funding_gap))}."))

    if df_monthly_cash["Trésorerie fin de mois"].min() < 0:
        comments.append(("risk", f"La trésorerie mensuelle devient négative sur l'année 1 avec un point bas de {fmt_mad(df_monthly_cash['Trésorerie fin de mois'].min())}."))
    else:
        comments.append(("good", "La trésorerie mensuelle reste positive sur l'ensemble de la première année."))

    if y1["Poids annuité / CA %"] > 0.20:
        comments.append(("risk", f"Le poids de l'annuité de dette représente {fmt_pct(y1['Poids annuité / CA %'])} du CA année 1, ce qui est élevé."))
    elif y1["Poids annuité / CA %"] > 0.10:
        comments.append(("warn", f"Le poids de l'annuité de dette représente {fmt_pct(y1['Poids annuité / CA %'])} du CA année 1, à surveiller."))
    else:
        comments.append(("good", "Le service de la dette reste maîtrisable au regard du chiffre d'affaires."))

    if break_even_df.loc[0, "Marge de sécurité %"] < 0.10:
        comments.append(("warn", "La marge de sécurité de l'année 1 est faible. Le niveau d'activité devra être sécurisé."))
    else:
        comments.append(("good", "La marge de sécurité du seuil de rentabilité est convenable."))

    if y5["Trésorerie fin d'année"] > y1["Trésorerie fin d'année"]:
        comments.append(("good", f"La trésorerie progresse entre l'année 1 ({treso1}) et l'année 5 ({treso5})."))
    else:
        comments.append(("warn", "La trésorerie reste tendue à moyen terme. Une réserve complémentaire serait prudente."))

    return comments

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## Paramètres généraux")
    project_name = st.text_input("Nom du projet", value="Clinique Multidisciplinaire")
    project_holder = st.text_input("Porteur du projet", value="Société CLINIQUE CAMILIA")
    legal_status = st.selectbox(
        "Statut juridique",
        ["SARL (IS)", "SAS (IS)", "SASU (IS)", "EURL (IS)", "Entreprise individuelle (IR)", "Autre"]
    )
    sector = st.selectbox(
        "Secteur",
        ["Clinique / Santé", "Cabinet médical", "Pharmacie", "Commerce", "Services", "Industrie légère", "Autre"]
    )
    city = st.text_input("Ville / commune", value="ERRAHMA")
    phone = st.text_input("Téléphone", value="")
    email = st.text_input("E-mail", value="")
    start_cash = st.number_input("Trésorerie de départ", min_value=0.0, value=500000.0, step=10000.0)
    income_tax_rate = st.number_input("Taux d'IS / impôt (%)", min_value=0.0, value=20.0, step=1.0) / 100
    inflation_rate = st.number_input("Inflation annuelle charges fixes (%)", min_value=0.0, value=3.0, step=0.5) / 100
    st.caption("Les sections de saisie alimentent automatiquement tous les tableaux financiers et le rapport expert.")

# =========================================================
# HERO
# =========================================================
st.markdown(
    f"""
    <div class="hero-card">
        <h1 style="margin-bottom:0.2rem;">{project_name}</h1>
        <p style="margin-top:0.2rem;">
        Outil professionnel d'étude financière et de business plan sur 5 ans :
        données à saisir, plan financier à imprimer, analyse de rentabilité,
        trésorerie mensuelle, structure de financement, rapport synthétique,
        export Excel multi-onglets et export PDF.
        </p>
        <div>
            <span class="legend-chip">Investissements & financements</span>
            <span class="legend-chip">Compte de résultat prévisionnel</span>
            <span class="legend-chip">SIG</span>
            <span class="legend-chip">BFR</span>
            <span class="legend-chip">Cash-flow</span>
            <span class="legend-chip">Rapport expert</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# TABS
# =========================================================
input_tab, print_tab, report_tab, export_tab = st.tabs([
    "1. Données à saisir",
    "2. Plan financier à imprimer",
    "3. Rapport synthétique",
    "4. Exports",
])

# =========================================================
# INPUT TAB
# =========================================================
with input_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">A. Identité du projet</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input("Nom / raison sociale", value=project_holder, disabled=True)
    with c2:
        st.text_input("Projet / activité", value=project_name, disabled=True)
    with c3:
        st.text_input("Ville / commune", value=city, disabled=True)
    st.markdown("<p class='small-note'>Ces informations sont reprises dans le rapport et dans les exports.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">B. Besoins de démarrage - montants TTC</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        establishment_cost = st.number_input("Frais d'établissement (TTC)", min_value=0.0, value=70000.0, step=5000.0)
        study_cost = st.number_input("Frais d'étude / ingénierie (TTC)", min_value=0.0, value=90000.0, step=5000.0)
        software_cost = st.number_input("Logiciels / formations (TTC)", min_value=0.0, value=180000.0, step=5000.0)
        brand_cost = st.number_input("Dépôt marque / brevet / modèle (TTC)", min_value=0.0, value=0.0, step=5000.0)
        entry_rights = st.number_input("Droits d'entrée (TTC)", min_value=0.0, value=0.0, step=5000.0)

    with c2:
        goodwill_cost = st.number_input("Achat fonds de commerce / parts (TTC)", min_value=0.0, value=0.0, step=10000.0)
        lease_right = st.number_input("Droit au bail (TTC)", min_value=0.0, value=0.0, step=10000.0)
        deposit_guarantee = st.number_input("Caution / dépôt de garantie", min_value=0.0, value=150000.0, step=10000.0)
        loan_fees = st.number_input("Frais de dossier", min_value=0.0, value=25000.0, step=5000.0)
        legal_fees = st.number_input("Frais de notaire / avocat", min_value=0.0, value=20000.0, step=5000.0)

    with c3:
        signage_cost = st.number_input("Enseigne / communication de lancement", min_value=0.0, value=40000.0, step=5000.0)
        real_estate_cost = st.number_input("Achat immobilier", min_value=0.0, value=0.0, step=10000.0)
        construction_cost = st.number_input("Construction / aménagement", min_value=0.0, value=1500000.0, step=10000.0)
        earthworks_cost = st.number_input("Terrassement", min_value=0.0, value=0.0, step=10000.0)
        equipment_cost = st.number_input("Matériel / équipements", min_value=0.0, value=2500000.0, step=10000.0)
        initial_stock = st.number_input("Stock initial", min_value=0.0, value=0.0, step=10000.0)

    st.markdown("<p class='small-note'>Saisis ici tous les investissements et frais de lancement. Ils alimentent les tableaux Investissements, Structure de financement, Amortissements et Cash-flow.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">C. Durées d\'amortissement (en années)</div>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        dur_establishment = st.number_input("Frais d'établissement", min_value=1, value=5)
        dur_study = st.number_input("Études", min_value=1, value=5)
        dur_software = st.number_input("Logiciels / formations", min_value=1, value=3)
        dur_brand = st.number_input("Marque / brevet", min_value=1, value=5)
    with a2:
        dur_entry = st.number_input("Droits d'entrée", min_value=1, value=5)
        dur_goodwill = st.number_input("Fonds de commerce / parts", min_value=1, value=10)
        dur_lease = st.number_input("Droit au bail", min_value=1, value=10)
        dur_loan_fees = st.number_input("Frais de dossier", min_value=1, value=5)
    with a3:
        dur_legal = st.number_input("Notaire / avocat", min_value=1, value=5)
        dur_signage = st.number_input("Communication / enseigne", min_value=1, value=4)
        dur_real_estate = st.number_input("Immobilier", min_value=1, value=20)
        dur_construction = st.number_input("Construction / aménagement", min_value=1, value=10)
    with a4:
        dur_earthworks = st.number_input("Terrassement", min_value=1, value=10)
        dur_equipment = st.number_input("Matériel / équipements", min_value=1, value=7)
        dur_stock = st.number_input("Stock initial", min_value=1, value=1)
        dur_deposit = st.number_input("Caution / dépôt", min_value=1, value=5)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">D. Financement du projet</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        equity = st.number_input("Apport personnel / capital social", min_value=0.0, value=1800000.0, step=10000.0)
        in_kind = st.number_input("Apports en nature", min_value=0.0, value=0.0, step=10000.0)
        shareholder_loan = st.number_input("Compte courant associé", min_value=0.0, value=0.0, step=10000.0)
    with f2:
        bank_loan_1 = st.number_input("Prêt bancaire n°1", min_value=0.0, value=2500000.0, step=10000.0)
        bank_loan_1_rate = st.number_input("Taux prêt n°1 (%)", min_value=0.0, value=6.0, step=0.25) / 100
        bank_loan_1_years = st.number_input("Durée prêt n°1 (ans)", min_value=1, value=7)
        moratorium_1_months = st.number_input("Différé prêt n°1 (mois)", min_value=0, value=0)
    with f3:
        bank_loan_2 = st.number_input("Prêt bancaire n°2", min_value=0.0, value=0.0, step=10000.0)
        bank_loan_2_rate = st.number_input("Taux prêt n°2 (%)", min_value=0.0, value=5.5, step=0.25) / 100
        bank_loan_2_years = st.number_input("Durée prêt n°2 (ans)", min_value=1, value=5)
        moratorium_2_months = st.number_input("Différé prêt n°2 (mois)", min_value=0, value=0)
        subsidy = st.number_input("Subvention", min_value=0.0, value=0.0, step=10000.0)
        other_financing = st.number_input("Autre financement", min_value=0.0, value=0.0, step=10000.0)
    st.markdown("<p class='small-note'>Montants annuels ou uniques selon la nature du financement. Les prêts alimentent automatiquement le tableau des emprunts, les charges financières et les cash-flows.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">E. Salaires et charges sociales</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        management_headcount = st.number_input("Nombre cadres / direction", min_value=0, value=1)
        management_monthly_salary = st.number_input("Salaire brut mensuel cadres", min_value=0.0, value=28000.0, step=1000.0)
        medical_headcount = st.number_input("Nombre personnel médical", min_value=0, value=3)
        medical_monthly_salary = st.number_input("Salaire brut mensuel personnel médical", min_value=0.0, value=12000.0, step=500.0)
    with s2:
        admin_headcount = st.number_input("Nombre personnel administratif", min_value=0, value=4)
        admin_monthly_salary = st.number_input("Salaire brut mensuel administratif", min_value=0.0, value=4500.0, step=500.0)
        support_headcount = st.number_input("Nombre personnel support", min_value=0, value=2)
        support_monthly_salary = st.number_input("Salaire brut mensuel support", min_value=0.0, value=4000.0, step=500.0)
    with s3:
        employer_social_rate = st.number_input("Taux charges sociales patronales (%)", min_value=0.0, value=21.09, step=0.1) / 100
        salary_growth_rate = st.number_input("Augmentation annuelle salaires (%)", min_value=0.0, value=3.0, step=0.5) / 100
        external_hr_cost = st.number_input("Autres coûts RH annuels", min_value=0.0, value=0.0, step=5000.0)

    st.markdown("<p class='small-note'>Les salaires doivent être saisis en mensuel brut. L'outil calcule automatiquement le brut annuel, les charges patronales et la masse salariale sur 5 ans.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">F. Charges fixes hors salaires - montants annuels année 1</div>', unsafe_allow_html=True)
    cfx1, cfx2, cfx3 = st.columns(3)
    with cfx1:
        insurance_y1 = st.number_input("Assurances (annuel)", min_value=0.0, value=60000.0, step=5000.0)
        telecom_y1 = st.number_input("Téléphone / internet (annuel)", min_value=0.0, value=40000.0, step=5000.0)
        subscriptions_y1 = st.number_input("Autres abonnements (annuel)", min_value=0.0, value=35000.0, step=5000.0)
        fuel_y1 = st.number_input("Carburant / transports (annuel)", min_value=0.0, value=45000.0, step=5000.0)
        travel_y1 = st.number_input("Déplacements / hébergement (annuel)", min_value=0.0, value=20000.0, step=5000.0)
        utilities_y1 = st.number_input("Eau / électricité / gaz (annuel)", min_value=0.0, value=180000.0, step=5000.0)
    with cfx2:
        mutual_y1 = st.number_input("Mutuelle (annuel)", min_value=0.0, value=30000.0, step=5000.0)
        supplies_y1 = st.number_input("Fournitures diverses (annuel)", min_value=0.0, value=45000.0, step=5000.0)
        maintenance_y1 = st.number_input("Entretien matériel / vêtements (annuel)", min_value=0.0, value=120000.0, step=5000.0)
        cleaning_y1 = st.number_input("Nettoyage des locaux (annuel)", min_value=0.0, value=50000.0, step=5000.0)
        marketing_y1 = st.number_input("Publicité / communication (annuel)", min_value=0.0, value=120000.0, step=5000.0)
        rent_y1 = st.number_input("Loyer et charges locatives (annuel)", min_value=0.0, value=750000.0, step=10000.0)
    with cfx3:
        leasing_y1 = st.number_input("Redevances crédit-bail (annuel)", min_value=0.0, value=0.0, step=5000.0)
        bank_fees_y1 = st.number_input("Frais bancaires (annuel)", min_value=0.0, value=20000.0, step=1000.0)
        taxes_y1 = st.number_input("Taxes (annuel)", min_value=0.0, value=40000.0, step=5000.0)
        accountant_y1 = st.number_input("Expert-comptable (annuel)", min_value=0.0, value=72000.0, step=5000.0)
        other_fixed_y1 = st.number_input("Autres charges fixes (annuel)", min_value=0.0, value=50000.0, step=5000.0)
    st.markdown("<p class='small-note'>Tous les montants de cette rubrique sont saisis en annuel pour l'année 1, puis revalorisés automatiquement avec l'inflation.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">G. Chiffre d\'affaires</div>', unsafe_allow_html=True)
    revenue_mode = st.radio(
        "Mode de saisie du CA",
        ["Saisie mensuelle année 1", "CA annuel + répartition mensuelle"],
        horizontal=True
    )

    rev1, rev2 = st.columns(2)
    with rev1:
        goods_sales_y1 = st.number_input("CA année 1 - vente de marchandises", min_value=0.0, value=0.0, step=10000.0)
        services_sales_y1 = st.number_input("CA année 1 - services", min_value=0.0, value=7064424.0, step=10000.0)
    with rev2:
        goods_growth_2 = st.number_input("Croissance marchandises année 2 (%)", value=0.0, step=1.0) / 100
        goods_growth_3 = st.number_input("Croissance marchandises année 3 (%)", value=0.0, step=1.0) / 100
        goods_growth_4 = st.number_input("Croissance marchandises année 4 (%)", value=10.0, step=1.0) / 100
        goods_growth_5 = st.number_input("Croissance marchandises année 5 (%)", value=0.0, step=1.0) / 100
        services_growth_2 = st.number_input("Croissance services année 2 (%)", value=70.0, step=1.0) / 100
        services_growth_3 = st.number_input("Croissance services année 3 (%)", value=50.0, step=1.0) / 100
        services_growth_4 = st.number_input("Croissance services année 4 (%)", value=30.0, step=1.0) / 100
        services_growth_5 = st.number_input("Croissance services année 5 (%)", value=20.0, step=1.0) / 100

    monthly_goods_values = []
    monthly_services_values = []

    st.markdown("<p class='small-note'>Cette répartition sert au budget de trésorerie mensuel de l'année 1.</p>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    cols = [m1, m2, m3, m4]
    for i, month in enumerate(MONTHS_SHORT):
        with cols[i % 4]:
            if revenue_mode == "Saisie mensuelle année 1":
                goods_val = st.number_input(f"{month} marchandises", min_value=0.0, value=0.0, step=1000.0, key=f"goods_{i}")
                services_val = st.number_input(f"{month} services", min_value=0.0, value=float(services_sales_y1 / 12), step=1000.0, key=f"services_{i}")
            else:
                goods_val = st.number_input(f"{month} % marchandises", min_value=0.0, value=float(100 / 12), step=0.5, key=f"goods_pct_{i}")
                services_val = st.number_input(f"{month} % services", min_value=0.0, value=float(100 / 12), step=0.5, key=f"services_pct_{i}")
            monthly_goods_values.append(goods_val)
            monthly_services_values.append(services_val)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">H. Charges variables et BFR</div>', unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    with v1:
        goods_purchase_rate = st.number_input("Coût d'achat marchandises / CA marchandises (%)", min_value=0.0, value=85.0, step=1.0) / 100
        service_variable_rate = st.number_input("Charges variables services / CA services (%)", min_value=0.0, value=20.0, step=1.0) / 100
        other_variable_rate = st.number_input("Autres variables / CA total (%)", min_value=0.0, value=3.0, step=0.5) / 100
    with v2:
        client_days = st.number_input("Crédit clients (jours)", min_value=0.0, value=45.0, step=1.0)
        supplier_days = st.number_input("Crédit fournisseurs (jours)", min_value=0.0, value=30.0, step=1.0)
        stock_days = st.number_input("Stock d'exploitation (jours)", min_value=0.0, value=30.0, step=1.0)
    with v3:
        safety_bfr_rate = st.number_input("Marge de sécurité BFR (%)", min_value=0.0, value=5.0, step=1.0) / 100
        vat_rate = st.number_input("TVA indicative (%)", min_value=0.0, value=20.0, step=1.0) / 100
        include_vat_buffer = st.checkbox("Inclure une réserve TVA dans le BFR", value=False)
    st.markdown("<p class='small-note'>Les délais clients, fournisseurs et stock servent au calcul du BFR d'exploitation. Le BFR de démarrage est calculé séparément selon une logique 3 mois.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# CALCULS
# =========================================================
need_items = [
    ("Frais d'établissement", establishment_cost, dur_establishment),
    ("Frais d'étude / ingénierie", study_cost, dur_study),
    ("Logiciels / formations", software_cost, dur_software),
    ("Dépôt marque / brevet / modèle", brand_cost, dur_brand),
    ("Droits d'entrée", entry_rights, dur_entry),
    ("Achat fonds de commerce / parts", goodwill_cost, dur_goodwill),
    ("Droit au bail", lease_right, dur_lease),
    ("Caution / dépôt de garantie", deposit_guarantee, dur_deposit),
    ("Frais de dossier", loan_fees, dur_loan_fees),
    ("Frais de notaire / avocat", legal_fees, dur_legal),
    ("Enseigne / communication", signage_cost, dur_signage),
    ("Achat immobilier", real_estate_cost, dur_real_estate),
    ("Construction / aménagement", construction_cost, dur_construction),
    ("Terrassement", earthworks_cost, dur_earthworks),
    ("Matériel / équipements", equipment_cost, dur_equipment),
    ("Stock initial", initial_stock, dur_stock),
]

capex_total = sum(v for _, v, _ in need_items)
resources_total = equity + in_kind + shareholder_loan + bank_loan_1 + bank_loan_2 + subsidy + other_financing

salary_rows = [
    ("Cadres / direction", management_headcount, management_monthly_salary),
    ("Personnel médical", medical_headcount, medical_monthly_salary),
    ("Personnel administratif", admin_headcount, admin_monthly_salary),
    ("Personnel support", support_headcount, support_monthly_salary),
]

salary_detail_rows = []
base_payroll_y1 = 0.0
base_gross_y1 = 0.0
base_social_y1 = 0.0

for label, hc, monthly_salary in salary_rows:
    annual_gross = hc * monthly_salary * 12
    annual_social = annual_gross * employer_social_rate
    annual_total = annual_gross + annual_social
    salary_detail_rows.append({
        "Poste": label,
        "Effectif": hc,
        "Salaire brut mensuel": monthly_salary,
        "Brut annuel": annual_gross,
        "Charges sociales": annual_social,
        "Coût annuel total": annual_total,
    })
    base_payroll_y1 += annual_total
    base_gross_y1 += annual_gross
    base_social_y1 += annual_social

base_payroll_y1 += external_hr_cost

salary_table = []
for year in YEARS:
    factor = (1 + salary_growth_rate) ** (year - 1)
    salary_table.append({
        "Année": year,
        "Salaires bruts": base_gross_y1 * factor,
        "Charges sociales": base_social_y1 * factor,
        "Autres coûts RH": external_hr_cost * factor,
        "Total salaires + charges": base_payroll_y1 * factor,
    })

df_salary = pd.DataFrame(salary_table)
df_salary_detail = pd.DataFrame(salary_detail_rows)

fixed_non_salary_y1 = {
    "Assurances": insurance_y1,
    "Téléphone / internet": telecom_y1,
    "Autres abonnements": subscriptions_y1,
    "Carburant / transports": fuel_y1,
    "Déplacements / hébergement": travel_y1,
    "Eau / électricité / gaz": utilities_y1,
    "Mutuelle": mutual_y1,
    "Fournitures diverses": supplies_y1,
    "Entretien matériel / vêtements": maintenance_y1,
    "Nettoyage des locaux": cleaning_y1,
    "Publicité / communication": marketing_y1,
    "Loyer et charges locatives": rent_y1,
    "Redevances crédit-bail": leasing_y1,
    "Frais bancaires": bank_fees_y1,
    "Taxes": taxes_y1,
    "Expert-comptable": accountant_y1,
    "Autres charges fixes": other_fixed_y1,
}

fixed_detail_years = []
for label, amount in fixed_non_salary_y1.items():
    row = {"Poste": label}
    for y in YEARS:
        row[f"Année {y}"] = amount * ((1 + inflation_rate) ** (y - 1))
    fixed_detail_years.append(row)
df_fixed_detail = pd.DataFrame(fixed_detail_years)

if revenue_mode == "Saisie mensuelle année 1":
    monthly_goods_y1 = np.array(monthly_goods_values, dtype=float)
    monthly_services_y1 = np.array(monthly_services_values, dtype=float)
    goods_sales_y1 = float(monthly_goods_y1.sum())
    services_sales_y1 = float(monthly_services_y1.sum())
else:
    goods_weights = normalize_distribution(monthly_goods_values)
    services_weights = normalize_distribution(monthly_services_values)
    monthly_goods_y1 = goods_weights * goods_sales_y1
    monthly_services_y1 = services_weights * services_sales_y1

revenue_goods = [goods_sales_y1]
revenue_services = [services_sales_y1]

for g in [goods_growth_2, goods_growth_3, goods_growth_4, goods_growth_5]:
    revenue_goods.append(revenue_goods[-1] * (1 + g))
for g in [services_growth_2, services_growth_3, services_growth_4, services_growth_5]:
    revenue_services.append(revenue_services[-1] * (1 + g))

total_ca = [g + s for g, s in zip(revenue_goods, revenue_services)]

loan_1_df = build_loan_schedule(bank_loan_1, bank_loan_1_rate, int(bank_loan_1_years), 5)
loan_2_df = build_loan_schedule(bank_loan_2, bank_loan_2_rate, int(bank_loan_2_years), 5)

df_loans = loan_1_df.copy()
df_loans["Mensualité"] = loan_1_df["Mensualité"] + loan_2_df["Mensualité"]
df_loans["Annuité"] = loan_1_df["Annuité"] + loan_2_df["Annuité"]
df_loans["Intérêts"] = loan_1_df["Intérêts"] + loan_2_df["Intérêts"]
df_loans["Remboursement capital"] = loan_1_df["Remboursement capital"] + loan_2_df["Remboursement capital"]
df_loans["Capital restant dû"] = loan_1_df["Capital restant dû"] + loan_2_df["Capital restant dû"]

amort_rows = []
total_amort_by_year = np.zeros(5)
for label, amount, duration in need_items:
    values = linear_amortization(amount, int(duration), 5)
    row = {"Poste": label, "Montant": amount, "Durée": duration}
    for idx, y in enumerate(YEARS):
        row[f"Année {y}"] = values[idx]
        total_amort_by_year[idx] += values[idx]
    amort_rows.append(row)
df_amort = pd.DataFrame(amort_rows)

goods_purchases = [g * goods_purchase_rate for g in revenue_goods]
service_variable = [s * service_variable_rate for s in revenue_services]
other_variable = [ca * other_variable_rate for ca in total_ca]
total_variable = [a + b + c for a, b, c in zip(goods_purchases, service_variable, other_variable)]

fixed_totals = []
for i, _year in enumerate(YEARS):
    factor = (1 + inflation_rate) ** i
    fixed_non_salary_total = sum(fixed_non_salary_y1.values()) * factor
    salary_total = df_salary.loc[i, "Total salaires + charges"]
    fixed_totals.append(fixed_non_salary_total + salary_total)

startup_purchases = (total_variable[0] / 12) * 3
startup_rent = (rent_y1 / 12) * 3
startup_payroll = (df_salary.loc[0, "Total salaires + charges"] / 12) * 3
startup_leasing = leasing_y1 / 12
bfr_startup_total = startup_purchases + startup_rent + startup_payroll + startup_leasing

bfr_values = []
for i in range(5):
    annual_purchases = total_variable[i]
    stock_component = annual_purchases * stock_days / 365
    client_component = total_ca[i] * client_days / 365
    supplier_component = annual_purchases * supplier_days / 365
    bfr = stock_component + client_component - supplier_component
    if include_vat_buffer:
        bfr += total_ca[i] * vat_rate * 0.15
    bfr *= (1 + safety_bfr_rate)
    bfr_values.append(bfr)

delta_bfr = [bfr_values[0]] + [bfr_values[i] - bfr_values[i - 1] for i in range(1, 5)]

global_rows = []
cash_balance = start_cash

for i, year in enumerate(YEARS):
    ca_goods = revenue_goods[i]
    ca_services = revenue_services[i]
    ca_total = total_ca[i]
    var_total = total_variable[i]
    gross_margin = ca_total - var_total
    fixed_cost = fixed_totals[i]
    ebitda = gross_margin - fixed_cost
    amort = total_amort_by_year[i]
    ebit = ebitda - amort
    financial = df_loans.loc[i, "Intérêts"]
    pre_tax = ebit - financial
    tax = max(pre_tax, 0.0) * income_tax_rate
    net_income = pre_tax - tax
    caf = net_income + amort
    principal = df_loans.loc[i, "Remboursement capital"]
    annuity = df_loans.loc[i, "Annuité"]

    initial_investment = capex_total if i == 0 else 0.0
    initial_bfr_startup = bfr_startup_total if i == 0 else 0.0
    initial_resources = resources_total if i == 0 else 0.0

    net_cash_flow = caf - delta_bfr[i] - principal - initial_investment - initial_bfr_startup + initial_resources
    cash_balance += net_cash_flow

    global_rows.append({
        "Année": year,
        "CA marchandises": ca_goods,
        "CA services": ca_services,
        "CA": ca_total,
        "Charges variables": var_total,
        "Marge brute": gross_margin,
        "Taux marge brute %": safe_div(gross_margin, ca_total),
        "Charges fixes": fixed_cost,
        "EBITDA": ebitda,
        "Marge EBITDA %": safe_div(ebitda, ca_total),
        "Amortissements": amort,
        "EBIT": ebit,
        "Charges financières": financial,
        "Résultat avant impôt": pre_tax,
        "Impôt": tax,
        "Résultat net": net_income,
        "Marge nette %": safe_div(net_income, ca_total),
        "CAF": caf,
        "BFR exploitation": bfr_values[i],
        "Variation BFR": delta_bfr[i],
        "Remboursement capital": principal,
        "Annuité dette": annuity,
        "Poids annuité / CA %": safe_div(annuity, ca_total),
        "Investissements initiaux": initial_investment,
        "BFR de démarrage": initial_bfr_startup,
        "Ressources initiales": initial_resources,
        "Flux net de trésorerie": net_cash_flow,
        "Trésorerie fin d'année": cash_balance,
    })

df_global = pd.DataFrame(global_rows)

tmcv = [max(1 - (total_variable[i] / max(total_ca[i], 1.0)), 0.0001) for i in range(5)]
seuil_rentabilite = [fixed_totals[i] / tmcv[i] for i in range(5)]

sig_rows = []
for i in range(5):
    commercial_margin = revenue_goods[i] - goods_purchases[i]
    production = revenue_services[i]
    value_added = df_global.loc[i, "Marge brute"] - (
        supplies_y1 * ((1 + inflation_rate) ** i)
        + telecom_y1 * ((1 + inflation_rate) ** i)
        + subscriptions_y1 * ((1 + inflation_rate) ** i)
    )
    ebe = df_global.loc[i, "EBITDA"]
    operating_result = df_global.loc[i, "EBIT"]
    current_result = df_global.loc[i, "Résultat avant impôt"]
    net_result = df_global.loc[i, "Résultat net"]

    sig_rows.append({
        "Année": YEARS[i],
        "Marge commerciale": commercial_margin,
        "Production de l'exercice": production,
        "Valeur ajoutée": value_added,
        "EBE": ebe,
        "Résultat d'exploitation": operating_result,
        "Résultat courant": current_result,
        "Résultat net": net_result,
    })
df_sig = pd.DataFrame(sig_rows)

df_invest_finance = pd.DataFrame(
    [{"Rubrique": label, "Montant": value} for label, value, _ in need_items]
    + [{"Rubrique": "TOTAL INVESTISSEMENTS", "Montant": capex_total}]
    + [
        {"Rubrique": "Apport personnel / capital social", "Montant": equity},
        {"Rubrique": "Apports en nature", "Montant": in_kind},
        {"Rubrique": "Compte courant associé", "Montant": shareholder_loan},
        {"Rubrique": "Prêt bancaire n°1", "Montant": bank_loan_1},
        {"Rubrique": "Prêt bancaire n°2", "Montant": bank_loan_2},
        {"Rubrique": "Subvention", "Montant": subsidy},
        {"Rubrique": "Autre financement", "Montant": other_financing},
        {"Rubrique": "TOTAL FINANCEMENT", "Montant": resources_total},
    ]
)

funding_plan_rows = []
for i in range(5):
    funding_plan_rows.append({
        "Année": YEARS[i],
        "Investissements": capex_total if i == 0 else 0.0,
        "BFR de démarrage": bfr_startup_total if i == 0 else 0.0,
        "BFR exploitation": bfr_values[i],
        "Variation BFR": delta_bfr[i],
        "CAF": df_global.loc[i, "CAF"],
        "Remboursement capital": df_global.loc[i, "Remboursement capital"],
        "Flux net": df_global.loc[i, "Flux net de trésorerie"],
        "Trésorerie fin d'année": df_global.loc[i, "Trésorerie fin d'année"],
    })
df_financing_plan = pd.DataFrame(funding_plan_rows)

funding_gap = resources_total - (capex_total + bfr_startup_total)
df_funding_structure = pd.DataFrame([
    {"Rubrique": "Apport personnel / capital social", "Montant": equity},
    {"Rubrique": "Apports en nature", "Montant": in_kind},
    {"Rubrique": "Compte courant associé", "Montant": shareholder_loan},
    {"Rubrique": "Prêt bancaire n°1", "Montant": bank_loan_1},
    {"Rubrique": "Prêt bancaire n°2", "Montant": bank_loan_2},
    {"Rubrique": "Subvention", "Montant": subsidy},
    {"Rubrique": "Autre financement", "Montant": other_financing},
    {"Rubrique": "TOTAL RESSOURCES", "Montant": resources_total},
    {"Rubrique": "TOTAL BESOINS (Investissements + BFR démarrage)", "Montant": capex_total + bfr_startup_total},
    {"Rubrique": "Excédent / déficit", "Montant": funding_gap},
])

charge_summary_rows = []
for i in range(5):
    factor = (1 + inflation_rate) ** i
    charge_summary_rows.append({
        "Année": YEARS[i],
        "Salaires + charges": df_salary.loc[i, "Total salaires + charges"],
        "Charges fixes hors salaires": sum(fixed_non_salary_y1.values()) * factor,
        "Charges variables": total_variable[i],
        "Amortissements": total_amort_by_year[i],
        "Charges financières": df_loans.loc[i, "Intérêts"],
        "Total charges": total_variable[i] + fixed_totals[i] + total_amort_by_year[i] + df_loans.loc[i, "Intérêts"],
    })
df_charge_summary = pd.DataFrame(charge_summary_rows)

revenue_summary_rows = []
for i in range(5):
    revenue_summary_rows.append({
        "Année": YEARS[i],
        "CA marchandises": revenue_goods[i],
        "CA services": revenue_services[i],
        "CA total": total_ca[i],
        "Croissance CA total %": 0.0 if i == 0 else safe_div(total_ca[i] - total_ca[i - 1], total_ca[i - 1]),
    })
df_revenue_summary = pd.DataFrame(revenue_summary_rows)

df_bfr_startup = pd.DataFrame([
    {"Rubrique": "Achats (3 mois)", "Montant": startup_purchases},
    {"Rubrique": "Loyers (3 mois)", "Montant": startup_rent},
    {"Rubrique": "Charges de personnel (3 mois)", "Montant": startup_payroll},
    {"Rubrique": "Redevance crédit-bail", "Montant": startup_leasing},
    {"Rubrique": "TOTAL BFR DE DÉMARRAGE", "Montant": bfr_startup_total},
])

df_bfr = pd.DataFrame({
    "Année": YEARS,
    "CA": total_ca,
    "Charges variables": total_variable,
    "BFR exploitation": bfr_values,
    "Variation BFR": delta_bfr,
})

monthly_total_sales = monthly_goods_y1 + monthly_services_y1
monthly_variable_total = (
    monthly_goods_y1 * goods_purchase_rate
    + monthly_services_y1 * service_variable_rate
    + monthly_total_sales * other_variable_rate
)

monthly_fixed_base = (sum(fixed_non_salary_y1.values()) + df_salary.loc[0, "Total salaires + charges"]) / 12
monthly_interest = df_loans.loc[0, "Intérêts"] / 12
monthly_principal = df_loans.loc[0, "Remboursement capital"] / 12
monthly_amort = total_amort_by_year[0] / 12

client_delay_months = int(round(client_days / 30))
supplier_delay_months = int(round(supplier_days / 30))

monthly_cash_rows = []
monthly_cash_balance = start_cash + resources_total - capex_total - bfr_startup_total

for i, month in enumerate(MONTHS):
    collected = monthly_total_sales[i] if client_delay_months == 0 else (monthly_total_sales[i - client_delay_months] if i - client_delay_months >= 0 else 0.0)
    paid_variable = monthly_variable_total[i] if supplier_delay_months == 0 else (monthly_variable_total[i - supplier_delay_months] if i - supplier_delay_months >= 0 else 0.0)

    fixed_paid = monthly_fixed_base
    interest_paid = monthly_interest
    principal_paid = monthly_principal

    net_monthly_cash = collected - paid_variable - fixed_paid - interest_paid - principal_paid
    monthly_cash_balance += net_monthly_cash

    monthly_cash_rows.append({
        "Mois": month,
        "Encaissements": collected,
        "Charges variables décaissées": paid_variable,
        "Charges fixes décaissées": fixed_paid,
        "Intérêts": interest_paid,
        "Remboursement capital": principal_paid,
        "Amortissements non décaissés": monthly_amort,
        "Flux net mensuel": net_monthly_cash,
        "Trésorerie fin de mois": monthly_cash_balance,
    })

df_monthly_cash = pd.DataFrame(monthly_cash_rows)

breakeven_df = pd.DataFrame({
    "Année": YEARS,
    "Charges fixes": fixed_totals,
    "Taux de marge sur coûts variables": tmcv,
    "Seuil de rentabilité": seuil_rentabilite,
    "Marge de sécurité": [total_ca[i] - seuil_rentabilite[i] for i in range(5)],
    "Marge de sécurité %": [safe_div(total_ca[i] - seuil_rentabilite[i], total_ca[i]) for i in range(5)],
})

diagnostic = generate_diagnostics(df_global, df_monthly_cash, funding_gap, breakeven_df)

# =========================================================
# PRINT TAB
# =========================================================
with print_tab:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Investissements initiaux", fmt_mad(capex_total))
    m2.metric("BFR de démarrage", fmt_mad(bfr_startup_total))
    m3.metric("CA année 1", fmt_mad(df_global.loc[0, "CA"]))
    m4.metric("Résultat net année 1", fmt_mad(df_global.loc[0, "Résultat net"]))
    m5.metric("Trésorerie fin année 5", fmt_mad(df_global.loc[4, "Trésorerie fin d'année"]))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Investissements et financements</div>', unsafe_allow_html=True)
        st.dataframe(money_styler(df_invest_finance, exclude=["Rubrique"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Structure de financement du projet</div>', unsafe_allow_html=True)
        st.dataframe(money_styler(df_funding_structure, exclude=["Rubrique"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Plan de financement sur 5 ans</div>', unsafe_allow_html=True)
        st.dataframe(money_styler(df_financing_plan, exclude=["Année"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Seuil de rentabilité économique</div>', unsafe_allow_html=True)
        st.dataframe(
            pct_money_styler(
                breakeven_df,
                pct_cols=["Taux de marge sur coûts variables", "Marge de sécurité %"],
                exclude=["Année"],
            ),
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Salaires et charges sociales</div>', unsafe_allow_html=True)
        st.dataframe(money_styler(df_salary, exclude=["Année"]), use_container_width=True)
        st.markdown("<p class='small-note'>Détail année 1</p>", unsafe_allow_html=True)
        st.dataframe(money_styler(df_salary_detail, exclude=["Poste", "Effectif"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Détail des amortissements</div>', unsafe_allow_html=True)
        st.dataframe(money_styler(df_amort, exclude=["Poste", "Durée"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c7, c8 = st.columns(2)
    with c7:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Synthèse des charges</div>', unsafe_allow_html=True)
        st.dataframe(money_styler(df_charge_summary, exclude=["Année"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c8:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Synthèse du chiffre d\'affaires</div>', unsafe_allow_html=True)
        st.dataframe(
            pct_money_styler(df_revenue_summary, pct_cols=["Croissance CA total %"], exclude=["Année"]),
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    c9, c10 = st.columns(2)
    with c9:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Compte de résultat prévisionnel sur 5 ans</div>', unsafe_allow_html=True)
        cr_cols = [
            "Année", "CA", "Charges variables", "Marge brute", "Charges fixes", "EBITDA",
            "Amortissements", "EBIT", "Charges financières", "Résultat avant impôt", "Impôt", "Résultat net"
        ]
        st.dataframe(money_styler(df_global[cr_cols], exclude=["Année"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c10:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Soldes intermédiaires de gestion</div>', unsafe_allow_html=True)
        st.dataframe(money_styler(df_sig, exclude=["Année"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c11, c12 = st.columns(2)
    with c11:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">BFR de démarrage</div>', unsafe_allow_html=True)
        st.dataframe(money_styler(df_bfr_startup, exclude=["Rubrique"]), use_container_width=True)
        st.markdown("<p class='small-note'>Logique inspirée du fichier : achats 3 mois + loyers 3 mois + charges de personnel 3 mois + redevance crédit-bail.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c12:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">BFR d\'exploitation</div>', unsafe_allow_html=True)
        st.dataframe(money_styler(df_bfr, exclude=["Année"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c13, c14 = st.columns(2)
    with c13:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Budget prévisionnel de trésorerie - année 1</div>', unsafe_allow_html=True)
        st.dataframe(money_styler(df_monthly_cash, exclude=["Mois"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c14:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Cash-flow du projet</div>', unsafe_allow_html=True)
        cashflow_df = df_global[
            [
                "Année", "CAF", "Variation BFR", "Remboursement capital",
                "Investissements initiaux", "BFR de démarrage", "Ressources initiales",
                "Flux net de trésorerie", "Trésorerie fin d'année"
            ]
        ].copy()
        st.dataframe(money_styler(cashflow_df, exclude=["Année"]), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Tableau des emprunts</div>', unsafe_allow_html=True)
    st.dataframe(money_styler(df_loans, exclude=["Année"]), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# REPORT TAB
# =========================================================
with report_tab:
    treso_fin_5 = fmt_mad(df_global.loc[4, "Trésorerie fin d'année"])

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Résumé exécutif</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
**Projet analysé :** {project_name}  
**Porteur :** {project_holder}  
**Secteur :** {sector}  
**Ville :** {city}  
**Statut juridique :** {legal_status}  
**Horizon étudié :** 5 ans  

Le projet mobilise **{fmt_mad(capex_total)}** d'investissements initiaux.  
Le **BFR de démarrage** est estimé à **{fmt_mad(bfr_startup_total)}**.  
Les ressources initiales ressortent à **{fmt_mad(resources_total)}**.  
Le chiffre d'affaires prévisionnel passe de **{fmt_mad(df_global.loc[0, 'CA'])}** en année 1 à **{fmt_mad(df_global.loc[4, 'CA'])}** en année 5.  
Le résultat net évolue de **{fmt_mad(df_global.loc[0, 'Résultat net'])}** en année 1 à **{fmt_mad(df_global.loc[4, 'Résultat net'])}** en année 5.  
La trésorerie de fin de période atteint **{treso_fin_5}**.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Diagnostic expert automatique</div>', unsafe_allow_html=True)
    for level, text in diagnostic:
        css = "good-box" if level == "good" else "warn-box" if level == "warn" else "risk-box"
        st.markdown(f"<div class='{css}'>{text}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    recommendations = []
    if funding_gap < 0:
        recommendations.append(f"Compléter le financement initial d'environ {fmt_mad(abs(funding_gap))} pour couvrir intégralement investissements et BFR de démarrage.")
    if df_monthly_cash["Trésorerie fin de mois"].min() < 0:
        recommendations.append("Prévoir une ligne de trésorerie court terme ou une trésorerie de départ plus élevée pour absorber les tensions de cash de l'année 1.")
    if df_global.loc[0, "Marge EBITDA %"] < 0.08:
        recommendations.append("Revoir la politique tarifaire, le niveau d'activité ou la structure de charges afin d'améliorer la rentabilité opérationnelle.")
    if df_global.loc[0, "Poids annuité / CA %"] > 0.15:
        recommendations.append("Négocier une durée d'endettement plus longue ou un différé afin d'alléger les sorties de cash au démarrage.")
    if breakeven_df.loc[0, "Marge de sécurité %"] < 0.10:
        recommendations.append("Sécuriser un volume minimal d'activité, car le seuil de rentabilité est proche du CA prévisionnel.")
    if not recommendations:
        recommendations = [
            "Le business plan paraît cohérent sur la base des hypothèses saisies.",
            "Actualiser le prévisionnel tous les trimestres à partir des réalisations pour piloter les écarts.",
            "Suivre mensuellement le BFR, les encaissements et la masse salariale pour sécuriser la trajectoire de trésorerie.",
        ]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Interprétation et recommandations</div>', unsafe_allow_html=True)
    for idx, reco in enumerate(recommendations, start=1):
        st.markdown(f"**{idx}.** {reco}")
    st.markdown('</div>', unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=df_global["Année"], y=df_global["CA"], name="CA"))
        fig1.add_trace(go.Scatter(x=df_global["Année"], y=df_global["Résultat net"], mode="lines+markers", name="Résultat net"))
        fig1.update_layout(title="Évolution du CA et du résultat net", template="plotly_dark", height=420)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_global["Année"], y=df_global["Trésorerie fin d'année"], mode="lines+markers", name="Trésorerie"))
        fig2.update_layout(title="Évolution de la trésorerie", template="plotly_dark", height=420)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    g3, g4 = st.columns(2)
    with g3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig3 = px.bar(df_bfr, x="Année", y=["BFR exploitation", "Variation BFR"], barmode="group", template="plotly_dark", title="BFR et variation du BFR")
        fig3.update_layout(height=420)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with g4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig4 = px.bar(
            df_charge_summary,
            x="Année",
            y=["Salaires + charges", "Charges fixes hors salaires", "Charges variables"],
            barmode="stack",
            template="plotly_dark",
            title="Structure des charges",
        )
        fig4.update_layout(height=420)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    g5, g6 = st.columns(2)
    with g5:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig5 = px.line(df_monthly_cash, x="Mois", y="Trésorerie fin de mois", template="plotly_dark", title="Trésorerie mensuelle année 1")
        fig5.update_layout(height=420)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with g6:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig6 = px.bar(df_revenue_summary, x="Année", y=["CA marchandises", "CA services"], barmode="stack", template="plotly_dark", title="Composition du chiffre d'affaires")
        fig6.update_layout(height=420)
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# EXPORT TAB
# =========================================================
with export_tab:
    excel_frames = {
        "Données synthèse": {"df": df_global, "pct_cols": ["Taux marge brute %", "Marge EBITDA %", "Marge nette %", "Poids annuité / CA %"]},
        "Investissements financements": {"df": df_invest_finance},
        "Structure financement": {"df": df_funding_structure},
        "Plan financement 5 ans": {"df": df_financing_plan},
        "Seuil rentabilité": {"df": breakeven_df, "pct_cols": ["Taux de marge sur coûts variables", "Marge de sécurité %"]},
        "Salaires": {"df": df_salary},
        "Détail salaires": {"df": df_salary_detail},
        "Charges fixes détail": {"df": df_fixed_detail},
        "Amortissements": {"df": df_amort},
        "Synthèse charges": {"df": df_charge_summary},
        "Synthèse CA": {"df": df_revenue_summary, "pct_cols": ["Croissance CA total %"]},
        "Compte résultat": {
            "df": df_global[
                ["Année", "CA", "Charges variables", "Marge brute", "Charges fixes", "EBITDA",
                 "Amortissements", "EBIT", "Charges financières", "Résultat avant impôt", "Impôt", "Résultat net"]
            ]
        },
        "SIG": {"df": df_sig},
        "BFR démarrage": {"df": df_bfr_startup},
        "BFR exploitation": {"df": df_bfr},
        "Trésorerie mensuelle": {"df": df_monthly_cash},
        "Cash-flow projet": {
            "df": df_global[
                ["Année", "CAF", "Variation BFR", "Remboursement capital", "Investissements initiaux",
                 "BFR de démarrage", "Ressources initiales", "Flux net de trésorerie", "Trésorerie fin d'année"]
            ]
        },
        "Emprunts": {"df": df_loans},
    }

    excel_file = make_excel_export(excel_frames)
    treso_fin_5_pdf = fmt_mad(df_global.loc[4, "Trésorerie fin d'année"])
    summary_text_pdf = (
        f"<b>Projet :</b> {project_name}<br/>"
        f"<b>Porteur :</b> {project_holder}<br/>"
        f"<b>Ville :</b> {city}<br/>"
        f"<b>Statut :</b> {legal_status}<br/>"
        f"<b>Investissements initiaux :</b> {fmt_mad(capex_total)}<br/>"
        f"<b>BFR de démarrage :</b> {fmt_mad(bfr_startup_total)}<br/>"
        f"<b>Ressources initiales :</b> {fmt_mad(resources_total)}<br/>"
        f"<b>CA année 1 :</b> {fmt_mad(df_global.loc[0, 'CA'])}<br/>"
        f"<b>CA année 5 :</b> {fmt_mad(df_global.loc[4, 'CA'])}<br/>"
        f"<b>Résultat net année 1 :</b> {fmt_mad(df_global.loc[0, 'Résultat net'])}<br/>"
        f"<b>Résultat net année 5 :</b> {fmt_mad(df_global.loc[4, 'Résultat net'])}<br/>"
        f"<b>Trésorerie fin année 5 :</b> {treso_fin_5_pdf}"
    )

    pdf_tables = {
        "Compte de résultat prévisionnel": {
            "df": df_global[
                ["Année", "CA", "Charges variables", "Marge brute", "Charges fixes", "EBITDA",
                 "Amortissements", "EBIT", "Charges financières", "Résultat avant impôt", "Impôt", "Résultat net"]
            ]
        },
        "Plan de financement 5 ans": {"df": df_financing_plan},
        "BFR de démarrage": {"df": df_bfr_startup},
        "BFR d'exploitation": {"df": df_bfr},
        "Trésorerie mensuelle année 1": {"df": df_monthly_cash},
        "Amortissements": {"df": df_amort[["Poste", "Montant", "Année 1", "Année 2", "Année 3", "Année 4", "Année 5"]]},
    }

    pdf_file = make_pdf_report(
        project_name=project_name,
        sector=sector,
        summary_text=summary_text_pdf,
        diagnostics=diagnostic,
        tables=pdf_tables,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Exports professionnels</div>', unsafe_allow_html=True)
    st.write("Télécharge le dossier financier complet en Excel multi-onglets ou en PDF de présentation.")
    st.download_button(
        "Télécharger l'étude financière en Excel",
        data=excel_file,
        file_name=f"{project_name.replace(' ', '_')}_etude_financiere.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "Télécharger le rapport PDF",
        data=pdf_file,
        file_name=f"{project_name.replace(' ', '_')}_rapport_financier.pdf",
        mime="application/pdf",
    )
    st.markdown("<p class='small-note'>requirements.txt : streamlit, pandas, numpy, plotly, xlsxwriter, reportlab, openpyxl</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div style="margin-top:20px; padding:14px; border-top:1px solid rgba(255,255,255,0.08); color:#cbd5e1;">
        Étude financière complète 5 ans - version Streamlit professionnelle
    </div>
    """,
    unsafe_allow_html=True,
)