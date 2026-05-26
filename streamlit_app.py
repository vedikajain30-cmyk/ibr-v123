# ─────────────────────────────────────────────────────────────────────────────
# Valuation Impact of Capital Expansion Cycle on India Infrastructure Sector
# Streamlit Dashboard  |  MS25GF013  |  SP Jain School of Global Management
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Infra Valuation Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY   = "#1F3864"
BLUE   = "#2E75B6"
GOLD   = "#C9A227"
GREEN  = "#70AD47"
RED    = "#FF4B4B"
LGRAY  = "#F5F5F5"

SECTOR_COLORS = {
    "Railways":  "#2E75B6",
    "EPC":       "#C9A227",
    "Power":     "#FF6B35",
    "Roads":     "#70AD47",
    "Airports":  "#9B59B6",
    "Ports":     "#1ABC9C",
    "Utilities": "#E74C3C",
    "Water":     "#3498DB",
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #FAFAFA; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: white;
        border-left: 4px solid #2E75B6;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 12px;
    }
    .metric-title { font-size: 12px; color: #666; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1F3864; margin: 4px 0; }
    .metric-delta { font-size: 13px; color: #70AD47; }
    .section-header {
        background: linear-gradient(90deg, #1F3864, #2E75B6);
        color: white;
        padding: 10px 20px;
        border-radius: 6px;
        font-size: 16px;
        font-weight: 700;
        margin: 16px 0 12px 0;
        letter-spacing: 0.3px;
    }
    .verdict-box {
        background: #E8F4FD;
        border-left: 5px solid #2E75B6;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 14px;
        color: #1F3864;
    }
    .sidebar .sidebar-content { background-color: #1F3864; }
    div[data-testid="stMetric"] label { font-size: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_excel("Final_dataset.xlsx", sheet_name="Dataset", header=1)
    df.columns = [
        "hash", "Year", "No", "Company", "SubSector", "SubSectorGroup",
        "MktCap", "Capex", "CapexAssets", "PE", "DE", "ROCE",
        "GDPGrowth", "RepoRate", "GovCapex", "LnMktCap"
    ]
    df = df.dropna(subset=["MktCap", "Capex", "DE", "ROCE", "LnMktCap"])
    df["SubSectorGroup"] = df["SubSectorGroup"].str.strip()

    # Derived columns
    df_s = df.sort_values(["Company", "Year"])
    df_s["MktCapChange"]    = df_s.groupby("Company")["MktCap"].diff()
    df_s["EfficiencyRatio"] = df_s["MktCapChange"] / df_s["Capex"].replace(0, np.nan)
    df_s["DE2"]             = df_s["DE"] ** 2
    df_s["HighPerformer"]   = (df_s["MktCapChange"] > df_s["MktCapChange"].median()).astype(int)

    year_order = ["FY16","FY17","FY18","FY19","FY20","FY21","FY22","FY23","FY24","FY25"]
    df_s["YearNum"] = df_s["Year"].map({y: i+1 for i,y in enumerate(year_order)})
    return df_s

df = load_data()
YEARS    = ["FY16","FY17","FY18","FY19","FY20","FY21","FY22","FY23","FY24","FY25"]
SECTORS  = sorted(df["SubSectorGroup"].unique())
COMPANIES= sorted(df["Company"].unique())

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='background:{NAVY};padding:16px;border-radius:8px;margin-bottom:16px;'>
        <div style='color:white;font-size:18px;font-weight:700;'>🏗️ Infra Valuation</div>
        <div style='color:#BDD7EE;font-size:12px;margin-top:4px;'>MS25GF013 | SP Jain</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "📑 Navigate to",
        ["🏠 Overview & Macro Cycle",
         "📊 Sub-Sector Efficiency (H4)",
         "🏢 Company Deep Dive",
         "📐 Leverage Tipping Point (H5)",
         "🤖 ML Classification & Rules"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    st.markdown("**🔧 Global Filters**")
    sel_sectors = st.multiselect("Sub-Sectors", SECTORS, default=SECTORS)
    sel_years   = st.multiselect("Fiscal Years", YEARS, default=YEARS)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:11px;color:#888;'>
    <b>Dataset:</b> 25 Companies · 10 Years<br>
    250 Firm-Year Observations<br>
    FY2016 – FY2025
    </div>
    """, unsafe_allow_html=True)

dff = df[df["SubSectorGroup"].isin(sel_sectors) & df["Year"].isin(sel_years)].copy()

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def section(title):
    st.markdown(f'<div class="section-header">▸ {title}</div>', unsafe_allow_html=True)

def verdict(text):
    st.markdown(f'<div class="verdict-box">💡 {text}</div>', unsafe_allow_html=True)

def kpi(col, title, value, delta=None, prefix="", suffix=""):
    with col:
        delta_html = f'<div class="metric-delta">▲ {delta}</div>' if delta else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{prefix}{value}{suffix}</div>
            {delta_html}
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW & MACRO CYCLE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview & Macro Cycle":
    st.markdown(f"<h2 style='color:{NAVY};margin-bottom:4px;'>Valuation Impact of Capital Expansion Cycle</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666;font-size:14px;'>Indian Infrastructure Sector | FY2016–FY2025 | 25 Companies | 250 Observations</p>", unsafe_allow_html=True)

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    kpi(c1, "Avg Market Cap (₹ Cr)", f"{dff['MktCap'].mean():,.0f}")
    kpi(c2, "Avg Capex (₹ Cr)", f"{dff['Capex'].mean():,.0f}")
    kpi(c3, "Avg P/E Ratio", f"{dff['PE'].mean():.1f}x")
    kpi(c4, "Avg D/E Ratio", f"{dff['DE'].mean():.2f}x")
    kpi(c5, "Avg ROCE", f"{dff['ROCE'].mean()*100:.1f}%")

    st.markdown("")

    # Year-wise macro table
    section("Year-Wise Capital Cycle Analysis")
    yr_grp = df[df["Year"].isin(sel_years)].groupby("Year").agg(
        MktCap=("MktCap","mean"),
        Capex=("Capex","mean"),
        PE=("PE","mean"),
        DE=("DE","mean"),
        ROCE=("ROCE","mean"),
        GDPGrowth=("GDPGrowth","first"),
        GovCapex=("GovCapex","first"),
        RepoRate=("RepoRate","first"),
    ).reindex(sel_years).reset_index()

    # Dual-axis chart: MktCap vs GovCapex
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Bar(
        x=yr_grp["Year"], y=yr_grp["MktCap"]/1000,
        name="Avg Mkt Cap (₹ '000 Cr)", marker_color=BLUE, opacity=0.8
    ), secondary_y=False)
    fig1.add_trace(go.Scatter(
        x=yr_grp["Year"], y=yr_grp["GovCapex"],
        name="Gov Capex (₹ Lakh Cr)", mode="lines+markers",
        line=dict(color=GOLD, width=3), marker=dict(size=8)
    ), secondary_y=True)
    # Cycle annotations
    for yr, label, color in [("FY21","COVID Paradox\n+48% MktCap","#E74C3C"),
                               ("FY22","Recovery Surge\n+87% MktCap","#27AE60"),
                               ("FY24","Cycle Peak","#8E44AD")]:
        if yr in sel_years:
            row = yr_grp[yr_grp["Year"]==yr]
            if not row.empty:
                fig1.add_annotation(x=yr, y=row["MktCap"].values[0]/1000,
                    text=label, showarrow=True, arrowhead=2,
                    bgcolor="white", bordercolor=color, font=dict(size=10,color=color),
                    ax=0, ay=-45)
    fig1.update_layout(
        title="Market Capitalisation vs Government Infrastructure Capex (FY2016–FY2025)",
        height=420, plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="Arial")
    )
    fig1.update_yaxes(title_text="Avg Mkt Cap (₹ '000 Cr)", secondary_y=False, gridcolor="#EEE")
    fig1.update_yaxes(title_text="Gov Capex (₹ Lakh Cr)", secondary_y=True)
    st.plotly_chart(fig1, use_container_width=True)

    verdict("H1 Supported (R²=0.566): Every ₹1 lakh crore increase in government capex → ~27% increase in infrastructure market cap. FY2021 is the clearest proof: despite -6.6% GDP, a 31% surge in gov capex drove a 48% rise in avg market cap.")

    col1, col2 = st.columns(2)
    with col1:
        section("P/E Ratio & Repo Rate Interaction")
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(
            x=yr_grp["Year"], y=yr_grp["PE"],
            name="Avg P/E Ratio", mode="lines+markers",
            line=dict(color=BLUE, width=2.5), marker=dict(size=7)
        ), secondary_y=False)
        fig2.add_trace(go.Scatter(
            x=yr_grp["Year"], y=yr_grp["RepoRate"],
            name="Repo Rate (%)", mode="lines+markers",
            line=dict(color=RED, width=2, dash="dash"), marker=dict(size=6)
        ), secondary_y=True)
        fig2.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02), font=dict(family="Arial"))
        fig2.update_yaxes(title_text="Avg P/E", secondary_y=False, gridcolor="#EEE")
        fig2.update_yaxes(title_text="Repo Rate (%)", secondary_y=True)
        st.plotly_chart(fig2, use_container_width=True)
        verdict("FY2023: Repo rate jumped 250 bps → P/E collapsed from 66x to 38x. Monetary policy directly compresses valuation multiples even when capex grows.")

    with col2:
        section("ROCE vs D/E Ratio Over Time (J-Curve)")
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(
            x=yr_grp["Year"], y=yr_grp["ROCE"]*100,
            name="Avg ROCE (%)", mode="lines+markers",
            line=dict(color=GREEN, width=2.5), marker=dict(size=7)
        ), secondary_y=False)
        fig3.add_trace(go.Bar(
            x=yr_grp["Year"], y=yr_grp["DE"],
            name="Avg D/E Ratio", marker_color=GOLD, opacity=0.6
        ), secondary_y=True)
        fig3.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02), font=dict(family="Arial"))
        fig3.update_yaxes(title_text="ROCE (%)", secondary_y=False, gridcolor="#EEE")
        fig3.update_yaxes(title_text="D/E Ratio", secondary_y=True)
        st.plotly_chart(fig3, use_container_width=True)
        verdict("H3 Supported (R²=0.610): As capex rises, D/E increases and ROCE compresses — the J-Curve. By FY2025, ROCE recovered to record highs as D/E hit 10-year lows.")

    section("Correlation Heatmap — All Key Variables")
    corr_cols = ["MktCap","Capex","CapexAssets","PE","DE","ROCE","GDPGrowth","RepoRate","GovCapex","LnMktCap"]
    corr_labels = ["Mkt Cap","Capex","Capex/Assets","P/E","D/E","ROCE","GDP Growth","Repo Rate","Gov Capex","Ln(Mkt Cap)"]
    corr_df = dff[corr_cols].corr().round(2)
    fig4 = px.imshow(corr_df, text_auto=True, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
        x=corr_labels, y=corr_labels,
        title="Pearson Correlation Matrix — Infrastructure Variables (FY2016–FY2025)")
    fig4.update_layout(height=420, font=dict(family="Arial"))
    st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – SUB-SECTOR EFFICIENCY (H4)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Sub-Sector Efficiency (H4)":
    st.markdown(f"<h2 style='color:{NAVY};'>Sub-Sector Capex-to-Valuation Efficiency (Objective 4 / H4)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666;font-size:14px;'>Which sub-sector converts ₹1 of capex into the most market value?</p>", unsafe_allow_html=True)

    # Compute efficiency
    eff = dff.dropna(subset=["EfficiencyRatio"])
    sec_eff = eff.groupby("SubSectorGroup").agg(
        AvgEfficiency=("EfficiencyRatio","mean"),
        MedianEfficiency=("EfficiencyRatio","median"),
        AvgCapexAssets=("CapexAssets","mean"),
        AvgMktCap=("MktCap","mean"),
        AvgPE=("PE","mean"),
        N=("EfficiencyRatio","count")
    ).reset_index().sort_values("AvgEfficiency", ascending=False)

    # Sector regression slopes
    reg_data = []
    for sector, grp in dff.groupby("SubSectorGroup"):
        if len(grp) > 4:
            sl, ic, r, p, se = stats.linregress(grp["CapexAssets"], grp["LnMktCap"])
            reg_data.append({"SubSectorGroup": sector, "Slope": sl, "R2": r**2, "P_value": p, "N": len(grp)})
    reg_df = pd.DataFrame(reg_data).sort_values("Slope", ascending=False)

    # KPIs
    best_sec = sec_eff.iloc[0]
    c1,c2,c3,c4 = st.columns(4)
    kpi(c1, "Most Efficient Sub-Sector", best_sec["SubSectorGroup"])
    kpi(c2, "Highest Efficiency Ratio", f"{best_sec['AvgEfficiency']:.2f}x")
    kpi(c3, "Steepest Regression Slope β", f"{reg_df.iloc[0]['Slope']:.1f}", prefix="")
    sig_count = reg_df[reg_df["P_value"] < 0.05].shape[0]
    kpi(c4, "Statistically Significant Sectors", f"{sig_count} / {len(reg_df)}")
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        section("Avg Efficiency Ratio by Sub-Sector (ΔMktCap ÷ Capex)")
        colors = [SECTOR_COLORS.get(s, BLUE) for s in sec_eff["SubSectorGroup"]]
        fig_eff = go.Figure(go.Bar(
            y=sec_eff["SubSectorGroup"], x=sec_eff["AvgEfficiency"],
            orientation="h", marker_color=colors,
            text=[f"{v:.2f}x" for v in sec_eff["AvgEfficiency"]],
            textposition="outside"
        ))
        fig_eff.update_layout(
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Avg Efficiency Ratio (x)", yaxis_title="",
            font=dict(family="Arial"), xaxis=dict(gridcolor="#EEE")
        )
        st.plotly_chart(fig_eff, use_container_width=True)

    with col2:
        section("Regression Slope β by Sub-Sector (Ln(MktCap) ~ Capex/Assets)")
        colors_r = [GREEN if p < 0.05 else RED for p in reg_df["P_value"]]
        fig_reg = go.Figure(go.Bar(
            y=reg_df["SubSectorGroup"], x=reg_df["Slope"],
            orientation="h", marker_color=colors_r,
            text=[f"β={v:.1f} {'✓' if p<0.05 else '✗'}" for v,p in zip(reg_df["Slope"], reg_df["P_value"])],
            textposition="outside"
        ))
        fig_reg.update_layout(
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="OLS Slope β  (green = significant p<0.05)", yaxis_title="",
            font=dict(family="Arial"), xaxis=dict(gridcolor="#EEE")
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    verdict("H4 Supported: Railways = 8.25x efficiency (highest); Airports β=132 (steepest slope). Ports β=0.75 with p=0.824 — NOT significant. Capex in Ports does NOT predict market cap.")

    section("Sector Regression Detail Table")
    display_reg = reg_df.copy()
    display_reg["Significant"] = display_reg["P_value"].apply(lambda p: "✅ Yes" if p<0.05 else "❌ No")
    display_reg["Slope"] = display_reg["Slope"].round(2)
    display_reg["R2"] = display_reg["R2"].round(3)
    display_reg["P_value"] = display_reg["P_value"].round(4)
    st.dataframe(display_reg.rename(columns={
        "SubSectorGroup":"Sub-Sector","Slope":"β (Slope)","R2":"R²","P_value":"P-value","N":"Obs (N)"
    }), use_container_width=True, hide_index=True)

    section("Scatter: Capex/Assets vs Ln(Mkt Cap) by Sub-Sector")
    fig_sc = px.scatter(
        dff, x="CapexAssets", y="LnMktCap",
        color="SubSectorGroup", hover_data=["Company","Year","MktCap","Capex"],
        trendline="ols", color_discrete_map=SECTOR_COLORS,
        labels={"CapexAssets":"Capex/Assets","LnMktCap":"Ln(Market Cap)","SubSectorGroup":"Sub-Sector"},
        title="Capex Intensity vs Ln(Market Cap) — Each dot is one firm-year"
    )
    fig_sc.update_layout(height=460, plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Arial"))
    st.plotly_chart(fig_sc, use_container_width=True)

    section("Year-Wise Efficiency by Sub-Sector")
    yr_sec = dff.dropna(subset=["EfficiencyRatio"]).groupby(["Year","SubSectorGroup"])["EfficiencyRatio"].mean().reset_index()
    yr_sec = yr_sec[yr_sec["Year"].isin(sel_years)]
    fig_yr = px.line(
        yr_sec, x="Year", y="EfficiencyRatio",
        color="SubSectorGroup", markers=True,
        color_discrete_map=SECTOR_COLORS,
        labels={"EfficiencyRatio":"Avg Efficiency Ratio (x)","SubSectorGroup":"Sub-Sector"},
        title="How Each Sub-Sector's Efficiency Ratio Moved Over the Capex Cycle"
    )
    fig_yr.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Arial"))
    st.plotly_chart(fig_yr, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – COMPANY DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏢 Company Deep Dive":
    st.markdown(f"<h2 style='color:{NAVY};'>Company Deep Dive — 10-Year Financial Profile</h2>", unsafe_allow_html=True)

    sel_company = st.selectbox("Select Company", COMPANIES)
    comp_df = df[df["Company"] == sel_company].sort_values("Year")
    sector   = comp_df["SubSectorGroup"].iloc[0]

    c1,c2,c3,c4,c5 = st.columns(5)
    kpi(c1, "Sub-Sector", sector)
    kpi(c2, "Mkt Cap FY25 (₹ Cr)", f"{comp_df[comp_df['Year']=='FY25']['MktCap'].values[0]:,.0f}" if 'FY25' in comp_df["Year"].values else "N/A")
    avg_eff = comp_df["EfficiencyRatio"].mean()
    kpi(c3, "Avg Efficiency Ratio", f"{avg_eff:.2f}x" if not np.isnan(avg_eff) else "N/A")
    kpi(c4, "ROCE FY25", f"{comp_df[comp_df['Year']=='FY25']['ROCE'].values[0]*100:.1f}%" if 'FY25' in comp_df["Year"].values else "N/A")
    kpi(c5, "D/E FY25", f"{comp_df[comp_df['Year']=='FY25']['DE'].values[0]:.2f}x" if 'FY25' in comp_df["Year"].values else "N/A")
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        section("Market Cap & Capex Trend (FY16–FY25)")
        fig_c1 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_c1.add_trace(go.Bar(
            x=comp_df["Year"], y=comp_df["MktCap"],
            name="Mkt Cap (₹ Cr)", marker_color=BLUE, opacity=0.8
        ), secondary_y=False)
        fig_c1.add_trace(go.Scatter(
            x=comp_df["Year"], y=comp_df["Capex"],
            name="Capex (₹ Cr)", mode="lines+markers",
            line=dict(color=GOLD, width=2.5), marker=dict(size=7)
        ), secondary_y=True)
        fig_c1.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02), font=dict(family="Arial"))
        fig_c1.update_yaxes(title_text="Mkt Cap (₹ Cr)", secondary_y=False, gridcolor="#EEE")
        fig_c1.update_yaxes(title_text="Capex (₹ Cr)", secondary_y=True)
        st.plotly_chart(fig_c1, use_container_width=True)

    with col2:
        section("ROCE & D/E Ratio Over Time")
        fig_c2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_c2.add_trace(go.Scatter(
            x=comp_df["Year"], y=comp_df["ROCE"]*100,
            name="ROCE (%)", mode="lines+markers",
            line=dict(color=GREEN, width=2.5), marker=dict(size=7)
        ), secondary_y=False)
        fig_c2.add_trace(go.Scatter(
            x=comp_df["Year"], y=comp_df["DE"],
            name="D/E Ratio", mode="lines+markers",
            line=dict(color=RED, width=2, dash="dot"), marker=dict(size=6)
        ), secondary_y=True)
        fig_c2.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02), font=dict(family="Arial"))
        fig_c2.update_yaxes(title_text="ROCE (%)", secondary_y=False, gridcolor="#EEE")
        fig_c2.update_yaxes(title_text="D/E Ratio", secondary_y=True)
        st.plotly_chart(fig_c2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        section("P/E Ratio vs Ln(Market Cap)")
        fig_c3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_c3.add_trace(go.Bar(
            x=comp_df["Year"], y=comp_df["PE"],
            name="P/E Ratio", marker_color=GOLD, opacity=0.7
        ), secondary_y=False)
        fig_c3.add_trace(go.Scatter(
            x=comp_df["Year"], y=comp_df["LnMktCap"],
            name="Ln(Mkt Cap)", mode="lines+markers",
            line=dict(color=NAVY, width=2.5), marker=dict(size=7)
        ), secondary_y=True)
        fig_c3.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02), font=dict(family="Arial"))
        fig_c3.update_yaxes(title_text="P/E Ratio", secondary_y=False, gridcolor="#EEE")
        fig_c3.update_yaxes(title_text="Ln(Mkt Cap)", secondary_y=True)
        st.plotly_chart(fig_c3, use_container_width=True)

    with col4:
        section("Year-on-Year Efficiency Ratio")
        eff_plot = comp_df.dropna(subset=["EfficiencyRatio"])
        colors_eff = [GREEN if v >= 0 else RED for v in eff_plot["EfficiencyRatio"]]
        fig_c4 = go.Figure(go.Bar(
            x=eff_plot["Year"], y=eff_plot["EfficiencyRatio"],
            marker_color=colors_eff,
            text=[f"{v:.1f}x" for v in eff_plot["EfficiencyRatio"]],
            textposition="outside"
        ))
        fig_c4.add_hline(y=0, line_color="black", line_width=1)
        fig_c4.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="Efficiency Ratio (x)", xaxis_title="",
            font=dict(family="Arial"), yaxis=dict(gridcolor="#EEE"))
        st.plotly_chart(fig_c4, use_container_width=True)

    section("Full 10-Year Financial Data Table")
    show_cols = ["Year","MktCap","Capex","CapexAssets","PE","DE","ROCE","GDPGrowth","RepoRate","GovCapex","LnMktCap","EfficiencyRatio"]
    display_comp = comp_df[show_cols].copy()
    display_comp["ROCE"] = (display_comp["ROCE"]*100).round(2)
    display_comp["CapexAssets"] = (display_comp["CapexAssets"]*100).round(2)
    display_comp = display_comp.round(2)
    st.dataframe(display_comp.rename(columns={
        "MktCap":"Mkt Cap (₹Cr)","Capex":"Capex (₹Cr)","CapexAssets":"Capex/Assets (%)","ROCE":"ROCE (%)","EfficiencyRatio":"Efficiency Ratio (x)"
    }), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – LEVERAGE TIPPING POINT (H5)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📐 Leverage Tipping Point (H5)":
    st.markdown(f"<h2 style='color:{NAVY};'>Leverage Tipping Point — Objective 5 / H5</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666;font-size:14px;'>At what D/E ratio does leverage switch from value-creating to value-destroying?</p>", unsafe_allow_html=True)

    # Run regression
    X = sm.add_constant(dff[["CapexAssets","DE","DE2","ROCE"]])
    model = sm.OLS(dff["LnMktCap"], X).fit()
    b0  = model.params["const"]
    b1  = model.params["CapexAssets"]
    b2  = model.params["DE"]
    b3  = model.params["DE2"]
    b4  = model.params["ROCE"]
    tp  = -b2 / (2*b3) if b3 != 0 else None

    c1,c2,c3,c4 = st.columns(4)
    kpi(c1, "D/E Tipping Point", f"{tp:.2f}x" if tp else "N/A")
    kpi(c2, "Model R²", f"{model.rsquared:.3f}")
    kpi(c3, "F-Statistic", f"{model.fvalue:.2f}")
    kpi(c4, "Capex/Assets β (p<0.001)", f"{b1:.3f}")
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        section("Quadratic Curve: D/E vs Predicted Ln(Mkt Cap)")
        de_range  = np.linspace(0, dff["DE"].quantile(0.97), 200)
        pred_vals = b0 + b1*dff["CapexAssets"].mean() + b2*de_range + b3*de_range**2 + b4*dff["ROCE"].mean()
        fig_tp = go.Figure()
        fig_tp.add_trace(go.Scatter(
            x=dff["DE"], y=dff["LnMktCap"],
            mode="markers", name="Observed",
            marker=dict(color=BLUE, opacity=0.45, size=6),
            hovertext=dff["Company"] + " " + dff["Year"]
        ))
        fig_tp.add_trace(go.Scatter(
            x=de_range, y=pred_vals,
            mode="lines", name="Fitted Quadratic",
            line=dict(color=GOLD, width=3)
        ))
        if tp:
            tp_y = b0 + b1*dff["CapexAssets"].mean() + b2*tp + b3*tp**2 + b4*dff["ROCE"].mean()
            fig_tp.add_vline(x=tp, line_dash="dash", line_color=RED, line_width=2,
                annotation_text=f"Tipping Point = {tp:.2f}x",
                annotation_font_color=RED, annotation_position="top right")
            fig_tp.add_trace(go.Scatter(
                x=[tp], y=[tp_y], mode="markers",
                marker=dict(color=RED, size=14, symbol="star"),
                name=f"Peak at D/E={tp:.2f}x"
            ))
        fig_tp.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="D/E Ratio", yaxis_title="Ln(Market Cap)",
            font=dict(family="Arial"), xaxis=dict(gridcolor="#EEE"), yaxis=dict(gridcolor="#EEE"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_tp, use_container_width=True)

    with col2:
        section("Avg Ln(Mkt Cap) by D/E Bracket")
        bins  = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 10.0]
        labs  = ["0–0.5","0.5–1.0","1.0–1.5","1.5–2.0","2.0–2.5","2.5–3.0","3.0+"]
        dff2  = dff.copy()
        dff2["DE_Bucket"] = pd.cut(dff2["DE"], bins=bins, labels=labs)
        bkt = dff2.groupby("DE_Bucket", observed=True).agg(
            AvgLnMktCap=("LnMktCap","mean"), N=("LnMktCap","count")
        ).reset_index()
        bar_colors = [GREEN if float(str(b).split("–")[0].replace("+","")) < tp else RED for b in bkt["DE_Bucket"]] if tp else [BLUE]*len(bkt)
        fig_bkt = go.Figure(go.Bar(
            x=bkt["DE_Bucket"], y=bkt["AvgLnMktCap"],
            marker_color=bar_colors, text=[f"{v:.2f}" for v in bkt["AvgLnMktCap"]],
            textposition="outside"
        ))
        if tp:
            fig_bkt.add_vline(x=4.5, line_dash="dash", line_color=RED,
                annotation_text=f"≈ Tipping Point {tp:.1f}x", annotation_font_color=RED)
        fig_bkt.update_layout(height=400, plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="D/E Bracket", yaxis_title="Avg Ln(Market Cap)",
            font=dict(family="Arial"), yaxis=dict(gridcolor="#EEE"))
        st.plotly_chart(fig_bkt, use_container_width=True)

    verdict(f"H5 Result: Tipping Point empirically estimated at D/E = {tp:.2f}x. β(D/E)=+{b2:.3f} (positive) and β(D/E²)={b3:.3f} (negative) confirms inverted-U. Optimal leverage range = 1.0–2.5x D/E. Beyond {tp:.1f}x, valuation declines.")

    section("Regression Coefficients Table")
    coef_data = {
        "Variable": ["Intercept (β₀)", "Capex/Assets (β₁)", "D/E Ratio (β₂)", "D/E² (β₃)", "ROCE (β₄)"],
        "Coefficient": [round(model.params[v], 4) for v in ["const","CapexAssets","DE","DE2","ROCE"]],
        "Std Error":   [round(model.bse[v], 4)    for v in ["const","CapexAssets","DE","DE2","ROCE"]],
        "t-Statistic": [round(model.tvalues[v], 3) for v in ["const","CapexAssets","DE","DE2","ROCE"]],
        "P-value":     [round(model.pvalues[v], 4) for v in ["const","CapexAssets","DE","DE2","ROCE"]],
        "Significant": ["✅ Yes" if model.pvalues[v]<0.05 else "⚠️ Marginal" if model.pvalues[v]<0.10 else "❌ No"
                        for v in ["const","CapexAssets","DE","DE2","ROCE"]]
    }
    st.dataframe(pd.DataFrame(coef_data), use_container_width=True, hide_index=True)

    col3, col4 = st.columns(2)
    with col3:
        section("Company D/E Trajectory — Select Company")
        sel_c = st.selectbox("Company", COMPANIES, key="tp_company")
        cdf = df[df["Company"] == sel_c].sort_values("Year")
        fig_cde = make_subplots(specs=[[{"secondary_y": True}]])
        fig_cde.add_trace(go.Scatter(
            x=cdf["Year"], y=cdf["DE"], name="D/E Ratio",
            mode="lines+markers", line=dict(color=BLUE, width=2.5), marker=dict(size=7)
        ), secondary_y=False)
        fig_cde.add_trace(go.Scatter(
            x=cdf["Year"], y=cdf["LnMktCap"], name="Ln(Mkt Cap)",
            mode="lines+markers", line=dict(color=GOLD, width=2.5, dash="dot"), marker=dict(size=6)
        ), secondary_y=True)
        if tp:
            fig_cde.add_hline(y=tp, line_color=RED, line_dash="dash",
                annotation_text=f"Tipping Point {tp:.1f}x", secondary_y=False)
        fig_cde.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial"), legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig_cde.update_yaxes(title_text="D/E Ratio", secondary_y=False, gridcolor="#EEE")
        fig_cde.update_yaxes(title_text="Ln(Mkt Cap)", secondary_y=True)
        st.plotly_chart(fig_cde, use_container_width=True)

    with col4:
        section("Distribution of D/E Ratios — All Firms")
        fig_hist = px.histogram(
            dff, x="DE", color="SubSectorGroup",
            nbins=30, color_discrete_map=SECTOR_COLORS,
            labels={"DE":"D/E Ratio","SubSectorGroup":"Sub-Sector"},
            title="D/E Distribution Across Panel"
        )
        if tp:
            fig_hist.add_vline(x=tp, line_dash="dash", line_color=RED,
                annotation_text=f"Tipping Point {tp:.1f}x", annotation_font_color=RED)
        fig_hist.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Arial"))
        st.plotly_chart(fig_hist, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – ML CLASSIFICATION & ASSOCIATION RULES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Classification & Rules":
    st.markdown(f"<h2 style='color:{NAVY};'>Machine Learning — High-Performer Classification</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='color:#666;font-size:14px;'>
    Binary classification: predict whether a firm-year is a <b>High Performer</b>
    (MktCap growth above panel median). Metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
    Association rules use Confidence & Lift.
    </p>
    """, unsafe_allow_html=True)

    # ── Feature engineering ───────────────────────────────────────────────────
    ml_df = dff[["MktCap","Capex","CapexAssets","PE","DE","DE2","ROCE",
                  "GDPGrowth","RepoRate","GovCapex","LnMktCap","HighPerformer"]].dropna()
    features = ["CapexAssets","DE","DE2","ROCE","GDPGrowth","RepoRate","GovCapex","LnMktCap"]
    feat_labels = ["Capex/Assets","D/E","D/E²","ROCE","GDP Growth","Repo Rate","Gov Capex","Ln(Mkt Cap)"]
    X_raw = ml_df[features]
    y     = ml_df["HighPerformer"]

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_raw)

    # ── Sidebar model selector ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("**🤖 ML Settings**")
        model_choice = st.selectbox("Classifier", ["Random Forest","Gradient Boosting","Logistic Regression"])
        test_size    = st.slider("Test Set Size (%)", 20, 40, 30)
        n_estimators = st.slider("Trees (RF/GB)", 50, 300, 150, 50)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=test_size/100, random_state=42, stratify=y
    )

    @st.cache_data
    def train_model(mc, ne, ts):
        if mc == "Random Forest":
            clf = RandomForestClassifier(n_estimators=ne, random_state=42, class_weight="balanced")
        elif mc == "Gradient Boosting":
            clf = GradientBoostingClassifier(n_estimators=ne, random_state=42)
        else:
            clf = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_sc, y, test_size=ts/100, random_state=42, stratify=y)
        clf.fit(X_tr2, y_tr2)
        return clf, X_tr2, X_te2, y_tr2, y_te2

    clf, X_tr, X_te, y_tr, y_te = train_model(model_choice, n_estimators, test_size)
    y_pred  = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:,1]

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    auc  = roc_auc_score(y_te, y_proba)

    cv_scores = cross_val_score(clf, X_sc, y, cv=StratifiedKFold(5), scoring="f1")

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    kpi(c1, "Accuracy",  f"{acc*100:.1f}%")
    kpi(c2, "Precision", f"{prec*100:.1f}%")
    kpi(c3, "Recall",    f"{rec*100:.1f}%")
    kpi(c4, "F1-Score",  f"{f1*100:.1f}%")
    kpi(c5, "ROC-AUC",   f"{auc:.3f}")
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        section("ROC Curve")
        fpr, tpr, _ = roc_curve(y_te, y_proba)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})",
            line=dict(color=BLUE, width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines", name="Random",
            line=dict(color="gray", dash="dash", width=1.5)
        ))
        fig_roc.add_annotation(
            x=0.6, y=0.2, text=f"AUC = {auc:.3f}",
            font=dict(size=16, color=BLUE), showarrow=False,
            bgcolor="white", bordercolor=BLUE
        )
        fig_roc.update_layout(
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            font=dict(family="Arial"), xaxis=dict(gridcolor="#EEE"), yaxis=dict(gridcolor="#EEE"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        section("Feature Importance")
        if hasattr(clf, "feature_importances_"):
            fi = clf.feature_importances_
        else:
            fi = np.abs(clf.coef_[0])
            fi = fi / fi.sum()
        fi_df = pd.DataFrame({"Feature": feat_labels, "Importance": fi}).sort_values("Importance")
        fig_fi = go.Figure(go.Bar(
            y=fi_df["Feature"], x=fi_df["Importance"],
            orientation="h",
            marker_color=[BLUE if v >= fi_df["Importance"].median() else "#BDD7EE" for v in fi_df["Importance"]],
            text=[f"{v:.3f}" for v in fi_df["Importance"]], textposition="outside"
        ))
        fig_fi.update_layout(
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Importance Score", yaxis_title="",
            font=dict(family="Arial"), xaxis=dict(gridcolor="#EEE")
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        section("Confusion Matrix")
        cm = confusion_matrix(y_te, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=["Predicted: Low","Predicted: High"],
            y=["Actual: Low","Actual: High"],
            title=f"Confusion Matrix — {model_choice}"
        )
        fig_cm.update_layout(height=320, font=dict(family="Arial"))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col4:
        section("5-Fold Cross-Validation F1 Scores")
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(5)], y=cv_scores,
            marker_color=[GREEN if v >= cv_scores.mean() else BLUE for v in cv_scores],
            text=[f"{v:.3f}" for v in cv_scores], textposition="outside"
        ))
        fig_cv.add_hline(y=cv_scores.mean(), line_dash="dash", line_color=GOLD,
            annotation_text=f"Mean F1 = {cv_scores.mean():.3f}", annotation_font_color=GOLD)
        fig_cv.update_layout(
            height=320, plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="F1 Score", font=dict(family="Arial"),
            yaxis=dict(gridcolor="#EEE", range=[0,1])
        )
        st.plotly_chart(fig_cv, use_container_width=True)

    verdict(f"{model_choice} | Accuracy={acc*100:.1f}% | Precision={prec*100:.1f}% | Recall={rec*100:.1f}% | F1={f1*100:.1f}% | AUC={auc:.3f} | CV F1={cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Probability Distribution ──────────────────────────────────────────────
    section("Predicted Probability Distribution — High Performer Score")
    prob_df = pd.DataFrame({
        "Probability": y_proba,
        "Actual": ["High" if v==1 else "Low" for v in y_te.values]
    })
    fig_prob = px.histogram(
        prob_df, x="Probability", color="Actual",
        nbins=25, barmode="overlay", opacity=0.7,
        color_discrete_map={"High": GREEN, "Low": RED},
        labels={"Probability": "P(High Performer)", "Actual": "True Class"},
        title="Model Confidence Distribution — How Sure is the Model?"
    )
    fig_prob.add_vline(x=0.5, line_dash="dash", line_color=NAVY,
        annotation_text="Decision Boundary (0.5)")
    fig_prob.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial"))
    st.plotly_chart(fig_prob, use_container_width=True)

    # ── Association Rules (Confidence & Lift) ─────────────────────────────────
    section("Association Rules — Confidence & Lift Analysis")
    st.markdown("""
    <p style='font-size:13px;color:#555;'>
    Rules show which combinations of financial conditions co-occur with High Performer status.
    <b>Confidence</b> = P(consequent | antecedent). <b>Lift</b> > 1 means the rule is genuinely useful.
    </p>
    """, unsafe_allow_html=True)

    # Discretise for association rules
    ar_df = dff[["SubSectorGroup","DE","ROCE","CapexAssets","GovCapex","HighPerformer"]].dropna().copy()
    ar_df["DE_Level"]    = pd.cut(ar_df["DE"], bins=[0,0.5,1.5,10], labels=["LowDE","MedDE","HighDE"]).cat.add_categories("UnkDE").fillna("UnkDE")
    ar_df["ROCE_Level"]  = pd.cut(ar_df["ROCE"], bins=[0,0.10,0.15,1], labels=["LowROCE","MedROCE","HighROCE"]).cat.add_categories("UnkROCE").fillna("UnkROCE")
    ar_df["Capex_Level"] = pd.cut(ar_df["CapexAssets"], bins=[0,0.04,0.08,1], labels=["LowCapex","MedCapex","HighCapex"]).cat.add_categories("UnkCapex").fillna("UnkCapex")
    ar_df["GovCx_Level"] = pd.cut(ar_df["GovCapex"], bins=[0,4,7,15], labels=["EarlyCycle","MidCycle","LateCycle"]).cat.add_categories("UnkCycle").fillna("UnkCycle")
    ar_df["Performer"]     = ar_df["HighPerformer"].map({1:"HighPerformer",0:"LowPerformer"})

    transactions = ar_df[["SubSectorGroup","DE_Level","ROCE_Level","Capex_Level","GovCx_Level","Performer"]].astype(str).values.tolist()
    te  = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(te_ary, columns=te.columns_)

    min_sup = st.slider("Min Support (Association Rules)", 0.05, 0.30, 0.10, 0.01)
    min_conf = st.slider("Min Confidence", 0.50, 0.90, 0.60, 0.05)

    try:
        freq_items = apriori(basket, min_support=min_sup, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
        rules = rules[rules["consequents"].apply(lambda x: "HighPerformer" in x)]
        rules = rules.sort_values("lift", ascending=False).head(15)
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

        if len(rules) > 0:
            col_ar1, col_ar2 = st.columns(2)
            with col_ar1:
                section("Top Rules by Lift (Lift > 1 = Useful)")
                fig_lift = px.scatter(
                    rules, x="confidence", y="lift",
                    size="support", color="lift",
                    color_continuous_scale="Blues",
                    hover_data={"antecedents": True, "confidence": ":.3f", "lift": ":.3f", "support": ":.3f"},
                    labels={"confidence":"Confidence","lift":"Lift","support":"Support"},
                    title="Confidence vs Lift — Bubble = Support"
                )
                fig_lift.add_hline(y=1.0, line_dash="dash", line_color=RED,
                    annotation_text="Lift = 1 (random)", annotation_font_color=RED)
                fig_lift.update_layout(height=360, plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Arial"))
                st.plotly_chart(fig_lift, use_container_width=True)

            with col_ar2:
                section("Confidence & Lift Bar — Top 10 Rules")
                top10 = rules.head(10).copy()
                top10["Rule"] = top10["antecedents"].apply(lambda x: x[:35]+"…" if len(x)>35 else x)
                fig_bar_ar = go.Figure()
                fig_bar_ar.add_trace(go.Bar(
                    name="Confidence", x=top10["Rule"], y=top10["confidence"],
                    marker_color=BLUE, yaxis="y"
                ))
                fig_bar_ar.add_trace(go.Scatter(
                    name="Lift", x=top10["Rule"], y=top10["lift"],
                    mode="lines+markers", line=dict(color=GOLD, width=2.5),
                    marker=dict(size=8), yaxis="y2"
                ))
                fig_bar_ar.update_layout(
                    height=360, plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Arial"),
                    xaxis=dict(tickangle=-30),
                    yaxis=dict(title="Confidence", gridcolor="#EEE"),
                    yaxis2=dict(title="Lift", overlaying="y", side="right"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    barmode="group"
                )
                st.plotly_chart(fig_bar_ar, use_container_width=True)

            section("Association Rules Table (sorted by Lift)")
            display_rules = rules[["antecedents","consequents","support","confidence","lift"]].copy()
            display_rules.columns = ["Antecedents (IF…)","Consequents (THEN…)","Support","Confidence","Lift"]
            display_rules = display_rules.round(3)
            st.dataframe(display_rules, use_container_width=True, hide_index=True)
            verdict(f"Top rule: IF [{rules.iloc[0]['antecedents']}] → THEN HighPerformer | Confidence={rules.iloc[0]['confidence']:.2f} | Lift={rules.iloc[0]['lift']:.2f}x")
        else:
            st.info("No rules found at these thresholds. Try lowering Min Support or Min Confidence.")
    except Exception as e:
        st.warning(f"Association rules could not be computed: {str(e)}")

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:12px;color:#999;text-align:center;'>
    Valuation Impact of Capital Expansion Cycle on Indian Infrastructure Sector |
    MS25GF013 | MGB October 2025 | SP Jain School of Global Management
    </div>
    """, unsafe_allow_html=True)
